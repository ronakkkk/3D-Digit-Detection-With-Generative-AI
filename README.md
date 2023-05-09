# 3D-Convolutional Neural Network for Digit Detection

## Dataset 
The dataset we are going to use is 3D-MNIST Dataset

### 3D-MNIST Dataset:
The 3D-MNIST dataset is a modified version of the original MNIST dataset, where each image is represented as a 3D volume instead of a 2D image. This dataset contains 60,000 training examples and 10,000 test examples, with each example consisting of a 32x32x32 voxel grid.

The dataset was created by converting the original MNIST dataset, which consists of handwritten digits, into a 3D voxel format. Each voxel has a binary value representing whether it is part of the digit or not. This dataset is useful for tasks that require 3D convolutional neural networks, such as 3D object recognition or segmentation.

The 3D-MNIST dataset is available on Kaggle and can be downloaded in either HDF5 or NumPy format. The dataset is split into a training set and a test set, with the training set containing 60,000 examples and the test set containing 10,000 examples.

In this repository, we provide a set of scripts for loading and preprocessing the 3D-MNIST dataset, as well as sample code for training and evaluating a 3D convolutional neural network on this dataset.

### Distribution of digits in the dataset:

![output](https://user-images.githubusercontent.com/37010825/237004108-9dfbbe1b-7980-4586-b1b2-39123e195953.png)

### 3D plot:

![3D Digit](https://user-images.githubusercontent.com/37010825/237004201-6c12b347-b62b-43dc-b100-28e515f48b52.PNG)

## Result
We are able to achieve an accuracy of 70% using 3D CNN model and following image show an classification report for the same.

![Capture](https://user-images.githubusercontent.com/37010825/237004472-03f05c93-63f6-412f-b235-90e96a53c211.PNG)

## Comparison:
The comparison is between MLP (50% accuracy) and 3D CNN (70% accuracy) using error plot.

MLP error plot:

![myplot](https://user-images.githubusercontent.com/37010825/237004978-c0a36d5e-0098-4287-9a9d-083c105ae148.png)

3D CNN error plot:

![myplot_cnn](https://user-images.githubusercontent.com/37010825/237005196-6603dcad-7146-41bb-a269-d9ec723d294b.png)



