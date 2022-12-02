import torch
import argparse
import numpy as np
import wandb
from einops import rearrange

from .train import arch_mapping, apply_to_vectors


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def load_checkpoint(path, info):
    filename = path.split('/')[-1]
    num_classes = 10
    if "dataset_cifar100" in filename:
        num_classes = 100
    if info[-1].endswith('.pth'):
        info[-1] = info[-1][:-4]
    model_name = info[0]
    print(path)
    model = arch_mapping[model_name](num_classes=num_classes).to(device)
    checkpoint = torch.load(path, map_location=device)
    if 'loss' in checkpoint.keys():
        print('loss:', checkpoint['loss'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    return model, checkpoint['ort_vectors']


def get_ready_for_svd(kernel, pad_to, strides):
    assert len(kernel.shape) in (4, 5)
    assert len(pad_to) == len(kernel.shape) - 2
    dim = len(pad_to)
    if isinstance(strides, int):
        strides = [strides] * dim
    else:
        assert len(strides) == dim
    for i in range(dim):
        assert pad_to[i] % strides[i] == 0
        assert kernel.shape[i] <= pad_to[i]
    old_shape = kernel.shape
    kernel_tr = torch.permute(kernel, dims=[dim, dim + 1] + list(range(dim)))
    padding_tuple = []
    for i in range(dim):
        padding_tuple.append(0)
        padding_tuple.append(pad_to[-i - 1] - kernel_tr.shape[-i - 1])
    kernel_pad = torch.nn.functional.pad(kernel_tr, tuple(padding_tuple))
    r1, r2 = kernel_pad.shape[:2]
    small_shape = []
    for i in range(dim):
        small_shape.append(pad_to[i] // strides[i])
    reshape_for_fft = torch.zeros((r1, r2, np.prod(np.array(strides))) + tuple(
        small_shape))
    if dim == 2:
        for i in range(strides[0]):
            for j in range(strides[1]):
                reshape_for_fft[:, :, i * strides[1] + j, :, :] = \
                    kernel_pad[:, :, i::strides[0], j::strides[1]]
    else:
        for i in range(strides[0]):
            for j in range(strides[1]):
                for k in range(strides[2]):
                    index = i * strides[1] * strides[2] + j * strides[2] + k
                    reshape_for_fft[:, :, index, :, :, :] = \
                        kernel_pad[:, :, i::strides[0], j::strides[1],
                        k::strides[2]]
    fft_results = torch.fft.fft2(reshape_for_fft).reshape(r1, -1, *small_shape)
    # sing_vals shape is (r1, 4r2, k, k, k)
    transpose_for_svd = np.transpose(fft_results, axes=list(range(2, dim + 2))
                                                       + [0, 1])
    # now the shape is (k, k, k, r1, 4r2)
    return kernel_pad, old_shape, r1, r2, small_shape, strides, \
           transpose_for_svd


def get_sing_vals(kernel, pad_to, stride):
    kernel = kernel.cpu().permute([2, 3, 0, 1])
    if kernel.shape[0] > pad_to[0]:
        k, n = kernel.shape[0], pad_to[0]
        assert k == n + 2 or k == n + 1
        pad_kernel = torch.nn.functional.pad(kernel, (0, 0, 0, 0,
                                             0, max(k, 2 * n) - k, 0,
                                                      max(k, 2 * n) - k))
        tmp = rearrange(pad_kernel,
                        '(w1 k1) (w2 k2) cin cout -> (k1 k2) (w1 w2) cin cout',
                        w1=2, w2=2)
        sv = torch.sqrt((tmp.sum(1) ** 2).sum(0))
        return sv
    before_svd = get_ready_for_svd(kernel, pad_to, stride)
    svdvals = torch.linalg.svdvals(before_svd[-1])[:, :, 0]
    return svdvals


def check_sing_vals(model, ort_vectors, index):
    with torch.no_grad():
        for child_name, child in model.named_children():
            if 'Conv' in child.__class__.__name__:
                vectors = ort_vectors[index]
                mean_norm, vector = apply_to_vectors(child, mean_norm, vectors)
                index += 1
                mean_norm /= len(vectors)
                svdvals = get_sing_vals(child.weight, vector.shape[1:],
                                        child.stride)
                max_sing_true = svdvals.max()
                print(index, mean_norm, max_sing_true)
                wandb.log({f"singular_values_{index}":
                           wandb.Histogram(max_sing_true)})
            else:
                index = check_sing_vals(child, ort_vectors, index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cp', required=True, type=str)
    args = parser.parse_args()

    model, ort_vectors = load_checkpoint(args.cp,
                                         args.cp.split('/')[-1].split('_'))
    wandb.init(
        project="ort_nla"
    )
    check_sing_vals(model, ort_vectors, 0)
