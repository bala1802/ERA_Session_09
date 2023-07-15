# Objective

Construct a Neural Network for the CIFAR10 dataset Multiclass classification, based on the below constraints

- The architecture should have 4 different blocks with 1 output block.
- Each block should have only 3 convolution layer.
- One of the layers must be Depthwise Seperable Convolution
- One of the layers must use Dilated Convolution
- Complusory Global Average Pooling post that Fully Connected Layer should be added
- The total Receptive fields used must be more than 44
- The data transformations should be done using `albumentations` library (`pip install albumentations`)
- Albumentations specifications:
  - Horizontal Flip
  - Shift Scale Rotate
  - Cutout
 - The `total params` used must be lesser than 200K
 - Accurcy should be `85%` with as many epochs as required
 
# How to read this repository?

The Session09.ipynb is the main notebook, inside which the `model.py` and `util.py` are used as helper class.

# `model.py`

This script is used to construct the Neural Network with below specifications

Architecture:

    - C1, C2, C3, C4
    - Each block will have 3 Convolution layers.
    - First layer will be Depthwise Seperable convolution layer; 
    - Second Layer will be a 1X1
    - Third layer will be a convolution layer with dilation as 2

Model Summary:

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]             288
            Conv2d-6          [-1, 128, 32, 32]           4,096
              ReLU-7          [-1, 128, 32, 32]               0
       BatchNorm2d-8          [-1, 128, 32, 32]             256
           Dropout-9          [-1, 128, 32, 32]               0
           Conv2d-10           [-1, 64, 16, 16]          73,728
           Conv2d-11           [-1, 64, 16, 16]             576
           Conv2d-12          [-1, 128, 16, 16]           8,192
             ReLU-13          [-1, 128, 16, 16]               0
      BatchNorm2d-14          [-1, 128, 16, 16]             256
          Dropout-15          [-1, 128, 16, 16]               0
           Conv2d-16          [-1, 128, 16, 16]           1,152
           Conv2d-17          [-1, 128, 16, 16]          16,384
             ReLU-18          [-1, 128, 16, 16]               0
      BatchNorm2d-19          [-1, 128, 16, 16]             256
          Dropout-20          [-1, 128, 16, 16]               0
           Conv2d-21             [-1, 64, 8, 8]          73,728
           Conv2d-22             [-1, 64, 8, 8]             576
           Conv2d-23             [-1, 64, 8, 8]           4,096
             ReLU-24             [-1, 64, 8, 8]               0
      BatchNorm2d-25             [-1, 64, 8, 8]             128
          Dropout-26             [-1, 64, 8, 8]               0
           Conv2d-27             [-1, 64, 8, 8]             576
           Conv2d-28             [-1, 64, 8, 8]           4,096
             ReLU-29             [-1, 64, 8, 8]               0
      BatchNorm2d-30             [-1, 64, 8, 8]             128
          Dropout-31             [-1, 64, 8, 8]               0
           Conv2d-32             [-1, 64, 6, 6]             576
           Conv2d-33             [-1, 64, 6, 6]           4,096
             ReLU-34             [-1, 64, 6, 6]               0
      BatchNorm2d-35             [-1, 64, 6, 6]             128
          Dropout-36             [-1, 64, 6, 6]               0
        AvgPool2d-37             [-1, 64, 1, 1]               0
           Conv2d-38             [-1, 10, 1, 1]             640
================================================================
Total params: 194,880
Trainable params: 194,880
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 8.18
Params size (MB): 0.74
Estimated Total Size (MB): 8.94
----------------------------------------------------------------

# Albumentations

- Albumentations is a popular library for image augmentation in machine learning and computer vision tasks.
- It offers a wide range of efficient augmentation techniques for tasks like object detection and image classification.
- Albumentations is fast, memory-efficient, and supports various input image formats.
- It allows customizable augmentation pipelines and can generate augmented images on the fly during training.
- It is particularly useful for reducing storage requirements and handling dynamic datasets.

## Albumentations techniques applied in this repository:

- `A.HorizontalFlip(p=0.5)`: flips an image horizontally with a 50% probability.
- `A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5)`: applies random shifts, scales, and rotations to an image with specified limits and a 50% probability.
- `A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=[0.4914, 0.4822, 0.4465], mask_fill_value = None, p=0.5),`: applies coarse dropout by randomly removing square-shaped regions from an image, with specified maximum and minimum hole sizes, fill values, and a 50% probability.
