# Adaptive Regularization
This is a work in progress.

## Instructions
Run `python ptb.py --reg <reg_arg>` where `<reg_arg>` is one of `none`, `static`, or `adaptive`. If static is selected, the network will be run with dropout, following the design in [Recurrent Neural Network Regularization](https://arxiv.org/pdf/1409.2329.pdf) by Zarembra et al. (2014)

## Coming Soon
- ArXiV paper
- Experiments on CIFAR-10/100 with LeNet, VGG and residual networks.
