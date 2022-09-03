# BYOL_pytorch
The repository provides an end-to-end implementation of self-supervised learning technique, i.e. BYOL, that leverages unlabeled image data to improve the performance of deep learning models in the domain of computer vision.

The implementation is based on the paper [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733) that is based on the principles of self-supervised learning. The BYOL approach takes a contrastive approach to contrastive learning suggesting that similar samples have similar representations. This approach has two main advantages, i.e. less sensitivity to systematic biases and efficient training as it does not rely on negative sampling.  
The data is trained on STL10 dataset.

Make sure to install the following packages before running the code:

kornia
pytorch_lightning
torch
