# Orthogonalization of Convolutions

This repository implements orthogonal regularization of convolutions. To check
the effectiveness of the regularizer, the fast computation of singular values
proposed in [a recent article](https://arxiv.org/abs/1805.10408) is
implemented. To learn more, see our
[presentation](https://docs.google.com/presentation/d/1WveaG0HqV2oxyJkNz0WCkN_v4qsoS0cSRwV3RQXuxro/edit?usp=sharing)
for project defense.

## Installation

In order to install all the necessary dependencies run the following command:

```
pip install -r requirements.txt
```

## Logging

The training code, as well as the code for computing singular values, performs 
logging to [Weights and Biases](https://wandb.ai). Upon the first run, please 
enter your wandb credentials, which can be obtained by registering a free 
account with the service.

## Training

### The dataset
We use Cifar-10 dataset. In our code it is downloaded via creating an instance
of a respective torchvision class (```torchvision.datasets.CIFAR10```). If you
already have this dataset on your machine in the suitable format for this 
class, then you might specify the path to it via ```--dataset-root``` argument.

### VGG
To train baseline VGG, run

```shell
python -m ort_nla_proj.train --dataset cifar10 --architecture vgg19 --epochs 140 --init-lr 0.01 --opt SGD --batch-size 128 --weight-dec 0.0001 --checkpoints-path checkpoints_retry --nesterov
```

In order to use the regularizer, one should specify the argument ```--orthogonal-k```
and set it to the value of the coefficient associated with the regularizer. The default
number of vectors used for regularization is 1, to change it, one should use the
option ```--num-of-vectors```. Other non-trivial options:

* ```--dist``` specifies the distribution of the random vectors. When it is set to "normal",
```--dist_mean``` and ```--dist_std``` are used to set mean and std of a normal distribution
* The option ```--log-ort-loss-by-layer``` will log orthogonal part of the loss
(i.e. the norms of the regularizers) on each training step for each layer in wandb, which might be
useful for debugging.
* The option ```--log-ort-loss-by-epoch``` will do two things. Firstly, it will 
turn on logging of the orthogonal part of the loss on each epoch (note the difference with the previous option)
for each layer in wandb. Secondly, it will turn on the collection of these losses
as a list (each element corresponds to one epoch) of dictionaries (each element
corresponds to one layer), which will be saved to the final checkpoint.
* The option ```--normalize-ort-by-layer``` will change the normalization of the regularizer.
By default, the sum of the norms of the vectors for each layer is divided by a constant equal to the 
sum of lengths of these vectors. However, with this option, the sum for each layer is normalized
independently by its own constant.

Using the above options, one can tune the training with the regularizer. The
most important options are learning rate (```--init-lr```) and coefficient of 
orthogonal loss (```--orthogonal-k```). An example of a training command:

```shell
python -m ort_nla_proj.train --dist rademacher --dataset cifar10 --architecture vgg19 --epochs 140 --init-lr 0.01 --opt SGD --batch-size 128 --nesterov --weight-dec 0.0001 --checkpoints-path checkpoints_nla --orthogonal-k 100000 --num-of-vectors 100 --log-ort-loss-by-epoch
```

This command creates a run like this: https://wandb.ai/whiteteadragon/ort_nla/runs/wjyb2fyd/.

### Working with checkpoints

Each training run creates two checkpoints: from the last epoch and from the best epoch with respect to validation accuracy.
To measure the singular values of the convolutional layers of the resulting network,
one should run the command

```shell
python -m ort_nla_proj.check_sing_vals --cp <path to cp>
```

It will print (and log to wandb) the pair (mean regularizer norm, maximal singular value) for each layer.
Moreover, the distribution of singular values for each layer will be logged to wandb.
This file has also an option ```--form-large-matrix <x>```, which will turn on the forming
of the matrix of a convolution on the layer x and printing of its Frobenius norm.
However, for some of the first layers this matrix is too large to form, so the code will crash.
Therefore, use it wisely.

## Obtained results

We have run experiments with different numbers of vectors, different coefficients of orthogonal loss
and different initial learning rates. Some graphs can be seen in our presentation linked above.
All the runs are logged to our [wandb](https://wandb.ai/whiteteadragon/ort_nla).
The results of the script ```check_sing_vals.py``` can be seen in our [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1Cls1-u_isUyawB9S2scH9uPS0gDlNyHuKaQy9796cOo/edit?usp=sharing) with results (it is a bit messy).