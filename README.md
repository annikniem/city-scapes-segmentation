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
   git clone git@github.com:annikniem/city-scapes-segmentation.git

## Results
These are the semantic segmentation results on the city-scapes dataset. Notice that the images have been down-sampled for performance reasons.

![Results](https://github.com/user-attachments/assets/4b3d3850-5df4-4e29-9485-0af3f384f585)

## References
[1]  Shelhamer E., Long J., and Darrell T., 2016, Fully Convolutional Networks for Semantic Segmentation, https://arxiv.org/pdf/1605.06211

[2] Cordts M., Omran M., Ramos S. et al., 2016, “The Cityscapes Dataset for Semantic Urban Scene Understanding”
