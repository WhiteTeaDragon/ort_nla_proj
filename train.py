import argparse
import os
import wandb
import torch
import pathlib
from torch import nn
from torchvision import transforms, datasets
from tqdm import tqdm
from time import time
from contextlib import redirect_stdout
import io
import torchsummary as ts

from .WideResNet import wide_resnet16_10
from .vgg import vgg19

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def apply_to_vectors(child, mean_norm, vectors):
    for i in range(len(vectors)):
        vector = vectors[i]
        x_p = torch.nn.functional.conv2d(vector, child.weight,
                                         stride=child.stride,
                                         padding=child.padding)
        x = torch.nn.functional.conv_transpose2d(
            x_p, child.weight, stride=child.stride,
            padding=child.padding)
        mean_norm += torch.linalg.norm(x - vector)
    return mean_norm, vector


def orthogonal_loss(model, ort_vectors, index):
    loss, shapes = 0, 0
    for child_name, child in model.named_children():
        if 'Conv' in child.__class__.__name__:
            vectors = ort_vectors[index]
            sum_norms, vector = apply_to_vectors(child, 0, vectors)
            index += 1
            loss += sum_norms
            shapes += 2 * child.weight.shape[0] ** 2
        else:
            loss_, shapes_, index = orthogonal_loss(child, ort_vectors, index)
            loss += loss_
            shapes += shapes_
    return loss, shapes, index


def count_acc(loader, criterion=nn.CrossEntropyLoss()):
    correct = 0
    total = 0
    total_loss = 0
    # since we're not training, we don't need to calculate the gradients for
    # our outputs
    model.eval()
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total, total_loss / len(loader)


classes_mapping = {
    'cifar10': [10, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
                datasets.CIFAR10],
    'cifar100': [100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761),
                 datasets.CIFAR100]
}

arch_mapping = {
    'wrn16-10': wide_resnet16_10,
    'vgg19': vgg19
}


def save_chp(epoch, model, optimizer, loss, args, ort_vectors, best=False):
    checkpoints_dir = pathlib.Path(args.checkpoints_path)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    if best:
        epoch = "best"
    filename = f"{args.architecture}_" \
               f"epochs_{args.epochs}_opt_{args.opt}_init_lr_{args.init_lr}_" \
               f"batch_size_{args.batch_size}_epoch_{epoch}_" \
               f"ol_{args.orthogonal_k}_" \
               f"dataset_{args.dataset}_num-of-vectors_{args.num_of_vectors}_" \
               f"dist_{args.dist}_mean_{args.dist_mean}_std_{args.dist_std}" \
               f".pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'ort_vectors': ort_vectors
    }, checkpoints_dir / filename)


def get_conv_output_shapes(model, shape=(3, 32, 32)):
    with io.StringIO() as buf, redirect_stdout(buf):
        ts.summary(model, shape)
        output = buf.getvalue()
    conv_lines = [line for line in output.split('\n') if 'Conv' in line or
                  'CiTT' in line]
    conv_outputs = [list(map(int, line.split('[')[1].split(']')[0].split(',')))
                    [1:] for line in conv_lines]
    return conv_outputs


def generate_random_vectors(model, input_shape, num_of_vectors, dist,
                            dist_mean, dist_std):
    conv_outputs = get_conv_output_shapes(model, input_shape)
    conv_inputs = [input_shape] + conv_outputs[:-1]
    random_vectors = []
    for i in range(len(conv_inputs)):
        if dist == 'uniform':
            random_vectors.append(torch.rand([num_of_vectors] + list(conv_inputs[i]
                                                                 )).to(device))
        elif dist == 'normal':
            standard_normal_dist = torch.randn([num_of_vectors] + list(conv_inputs[i]))
            normal_dist = standard_normal_dist * dist_std + dist_mean
            random_vectors.append(normal_dist.to(device))
        elif dist == 'rademacher':
            uniform_dist = torch.rand([num_of_vectors] + list(conv_inputs[i]))
            rademacher_dist = torch.where(uniform_dist < 0.5, -1., 1.)
            random_vectors.append(rademacher_dist.to(device))
        else:
            raise Exception("Specify correct vector distribution: --dist <>")
    return random_vectors


if __name__ == '__main__':
    SEED = 3407
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)

    """### Parsing arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--init-lr', default=0.01, type=float)
    parser.add_argument('--architecture', default='wrn16-10',
                        choices=arch_mapping.keys())
    parser.add_argument('--opt', default="SGD", choices=["Adam", "SGD"])
    parser.add_argument('--weight-dec', default=1e-4, type=float)
    parser.add_argument('--nesterov', dest='nesterov', action='store_true')
    parser.add_argument('--no-nesterov', dest='nesterov', action='store_false')
    parser.add_argument('--save_chp_every', default=200, type=int)
    parser.add_argument('--checkpoints-path', default="/home/checkpoints",
                        type=str)
    parser.add_argument('--dataset-root', default='./data', type=str)
    parser.add_argument('--orthogonal-k', default=-1, type=float)
    parser.add_argument('--num-of-vectors', default=1, type=int)
    parser.add_argument('--dist', default='uniform', type=str)
    parser.add_argument('--dist_mean', default=0, type=int)
    parser.add_argument('--dist_std', default=1, type=int)

    parser.set_defaults(nesterov=False)

    args = parser.parse_args()

    """### Loading Data"""

    wandb.init(
        project="ort_nla"
    )

    batch_size = args.batch_size
    num_classes, cifar_mean, cifar_std, dataset = classes_mapping[args.dataset]
    if args.architecture == 'vgg19':
        cifar_mean, cifar_std = 0.5, 0.5

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    # load the data.
    trainset = dataset(root=args.dataset_root, train=True,
                                download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               pin_memory=True,
                                               shuffle=True, num_workers=2)

    testset = dataset(root=args.dataset_root, train=False,
                               download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              pin_memory=True,
                                              shuffle=False, num_workers=2)

    epochs = args.epochs

    """#### Model without clipping"""

    kwargs = {}
    if 'wrn' in args.architecture:
        kwargs["num_classes"] = num_classes
        model = arch_mapping[args.architecture](**kwargs).to(device)
    elif 'vgg' in args.architecture:
        model = arch_mapping[args.architecture](num_classes=num_classes,
                                                **kwargs).to(device)
    else:
        model = arch_mapping[args.architecture](num_classes=num_classes,
                                                pretrained=False).to(device)
    model_kwargs = {}
    print(model)
    print(sum(p.numel() for p in model.parameters()))

    parameters = model.parameters()

    if args.opt == "Adam":
        if "vgg" in args.architecture:
            optimizer = torch.optim.Adam(parameters, lr=args.init_lr,
                                         amsgrad=True, eps=1e-7)
        else:
            optimizer = torch.optim.Adam(parameters, lr=args.init_lr)
    else:
        optimizer = torch.optim.SGD(parameters, lr=args.init_lr,
                                    weight_decay=args.weight_dec,
                                    momentum=0.9, nesterov=args.nesterov)
    lr_steps = epochs * len(train_loader)
    if 'wrn' in args.architecture:
        assert args.epochs in [1, 200, 400]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[0.3 * args.epochs * len(train_loader),
                                   0.6 * args.epochs * len(train_loader),
                                   0.8 * args.epochs * len(train_loader)],
            gamma=0.2)
    elif 'vgg' in args.architecture:
        assert args.epochs in [1, 140]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[(100 / 140) * args.epochs * len(train_loader),
                        (120 / 140) * args.epochs * len(train_loader)],
            gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[lr_steps // 2, (3 * lr_steps) // 4],
            gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    count = 0
    orthogonal = (args.orthogonal_k > 0)

    best_val_acc = 0
    ort_vectors = generate_random_vectors(model, trainset[0][0].shape,
                                          args.num_of_vectors, args.dist,
                                          args.dist_mean, args.dist_std)
    for epoch in range(epochs):
        running_loss = 0.0
        running_orthogonal_loss = 0.0
        opening = time()
        model.train()
        for i, (X, y) in tqdm(enumerate(train_loader),
                              total=len(train_loader)):
            count += 1
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            output = model(X)
            loss = criterion(output, y)
            if orthogonal:
                orthogonal = True
                loss_, shapes_, _ = orthogonal_loss(model, ort_vectors, 0)
                running_orthogonal_loss += (loss_ / shapes_).item()
                loss += (args.orthogonal_k / shapes_) * loss_
            loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
        training_end = time()

        lr = scheduler.get_last_lr()[0]
        val_acc, val_loss = count_acc(test_loader)
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            save_chp(epoch, model, optimizer, loss, args, ort_vectors, best=True)
        wandb.log({'loss': running_loss / len(train_loader),
                   'orthogonal_loss':
                       running_orthogonal_loss / len(train_loader),
                   'acc': count_acc(train_loader)[0], 'val_acc': val_acc,
                   'val_loss': val_loss, 'epoch': epoch, 'lr': lr,
                   'time': time() - opening, 'val_time': time() - training_end,
                   'train_time': training_end - opening})
        print('%d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))
        print(f'train acc: {count_acc(train_loader)}')
        if epoch != 0 and epoch % args.save_chp_every == 0:
            save_chp(epoch, model, optimizer, loss, args, ort_vectors)

    print(f'test acc: {count_acc(test_loader)}')

    save_chp(epoch, model, optimizer, loss, args, ort_vectors)
