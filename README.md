# NIFF (Neural Implicit Frequency Filters)

Code for our work [As large as it gets – Studying Infinitely Large Convolutions via Neural Implicit Frequency Filters](https://openreview.net/forum?id=xRy1YRcHWj) accepted at TMLR with **Featured Certification** and presented at ICLR 2025 in Singapore.
We include several convolutional neural network (CNN) architectures incorporating our Neural Implicit Frequency Filters (NIFF) and the Code of our NIFF.

We introduce a new tool, NIFF, to study the effective filter size learned by CNNs. Our NIFF inherently solves the following challenges:
- we need an **effective** means to **train** models with **large filters** (potentially as large as the input data) **without increasing the number of learnable parameters**
- the employed convolution operation should be a **plug-and-play module** that can replace conventional convolutions in a CNN and allow for an efficient implementation in current frameworks
- the study of filter sizes has to be **decoupled from other aspects** such as the network width or the number of learnable parameters
- the **cost** of the convolution operation itself has to **remain manageable** i.e.~we can not na\"{\i}vely increase the size of the convolution kernel

## Training and Evaluation 

The training script can be found at https://github.com/facebookresearch/ConvNeXt.

To visualize our NIFF's spatial learned filters, use the following [Notebook](https://github.com/GeJulia/NIFF/blob/main/show_spatial_kernel.ipynb).

To evaluate the kernel mass ratio of the learned filter, please use this [Notebook](https://github.com/GeJulia/NIFF/blob/main/plot_kernel_mass_ratio.ipynb).


Network structures are taken from  https://github.com/pytorch/vision/blob/main/torchvision/models and https://github.com/facebookresearch/ConvNeXt.


## Citation

Would you like to reference our **`NIFF`**? 

Then consider citing our [paper](https://openreview.net/forum?id=xRy1YRcHWj):

```bibtex
@inproceedings{grabinski2024niff,
  title     = {As large as it gets – Studying Infinitely Large Convolutions via Neural Implicit Frequency Filters},
  author    = {Grabinski, Julia and Keuper, Janis and Keuper, Margret},
  booktitle = {TMLR},
  year      = {2024},
  url       = {https://openreview.net/forum?id=xRy1YRcHWj}
}
```
