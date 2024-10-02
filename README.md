# Semantic segmentation of the city-scapes dataset using FCN-8
In this repository you can find a PyTorch implementation of the FCN-8 model along with training and 
validation algorithms as well as all related data processing. 
FCN-8 is one of the first machine learning -based semantic segmentation algorithms.
FCN-8 is a fully convolutional encoder-decoder model that utilises skip-connections
for improved segmentation results. [1]

## Repository contents
This repository contains the following:
- Src:
     - FCN-8 model architecture.
     - All  code required to train and validate the FCN-8 model.
     - All related data processing required for the city-scapes dataset.
- Figures:
     - All relevant figures such as results, data examples and loss curves.

## Installation
1. Clone the repository:
   git clone git@github.com:annikniem/city-scapes.segmentation.git

## Results
These are the semantic segmentation results on the city-scapes dataset. Notice that the images have been down-sampled for performance reasons.


## References
[1] Mardani, M., Sun, Q., Vasanawala, S. et al., 2018, Neural Proximal Gradient Descent for Compressive Imaging, https://arxiv.org/pdf/1806.03963

[2] Facebook and NYU FastMRI knee dataset, 2020, https://github.com/facebookresearch/fastMRI
