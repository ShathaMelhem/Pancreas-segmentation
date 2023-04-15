# Pancreas-segmentation

# Introduction

This is the code repository for the abstract of our new Mathitís U-Net.In our work, we inspired by Resnet and Inception principles.
Mathitís U-Net containes the main component of  Mathitís block. this block was added to enhance the performance of the original U-net in terms of DSC and provide a robust 
version of unet that proper for pancreas segmentation.

# Main Dependencies
google colab dependencies 

# File List

Folder | File	Description
-------|----------------
README.txt|	the README file
DATA2NPY|	codes to transfer the NIH dataset into NPY format
dicom2npy.py	|transferring image data (DICOM) into NPY format
nii2npy.py|	transferring label data (NII) into NPY format
slice.py| convert each 3D CT image into number of 2D slices for the axial plane ONLY 
pslice.py| convert each 3D CT image into number of 2D slices for the axial, sagittal, and coronal planes
data.py	|the data layer to 1. create train & test input to Network as numpy arrays 2. Load the train & test numpy arrays.
Mathitís U-Net.py| code to build and train the 2D Mathitís U-Net
pipeline |executable script
init.py	|the initialization functions
utils.py|	the common functions

# References
[1]https://github.com/snapfinger/pancreas-seg
[2] Y. Zhou, L. Xie, W. Shen, Y. Wang, E. Fishman and A. Yuille, "A Fixed-Point Model for Pancreas Segmentation in Abdominal CT Scans", Proc. MICCAI, 2017

[3] H. Roth, L. Lu, A Farag, H-C Shin, J Liu, E. Turkbey, and R. M. Summers, "DeepOrgan: Multi-level deep convolutional networks for automated pancreas segmentation", Proc. MICCAI, 2015.

[4] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", Proc. MICCAI, 2015.

[5] Liu, Yijun, and Shuang Liu. "U-net for pancreas segmentation in abdominal CT scans." IEEE international symposium on biomedical imaging. Vol. 2018. 2018.
 
