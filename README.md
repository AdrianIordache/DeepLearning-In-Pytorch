# Deep Learning In Pytorch

Those projects are done as part of the course held by Bitdefender researchers.

## Bucharest Housing Classification And Regression

In this introductory project will use some basic analytics to develop intuition about the problem. Will try to achieve good results with both strategies (Classification and Regression), explaining in detail which things do or do not work on this dataset.

 
This project uses [scikit-learn](https://scikit-learn.org/stable/) for establishing baseline models using tree-based methods like [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) or non-tree based methods like [MLP (Multi-layer Perceptron)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html), which represent the neural networks for scikit-learn.

And after that will try to reach the same or better results in [Pytorch](https://pytorch.org/) with some basic models.

## Transfer Learning On Counting MNIST Dataset

In the second project will explore concepts like fully convolutional neural networks and transfer learning.

Will train a standard CNN on the MNIST dataset (28x28 px) and use those weights in a fully convolutional neural network as prior knowledge for predicting the number of digits in a larger image (100x100 px) from Counting MNIST. This experiment will be done without training the model for this particular task, only by fine-tuning the thresholds.

Will compare these results by training a CNN model on the Counting MNIST. 


## Google Landmark Recognition Challenge
This will be the final project of the course which represents a comparative analysis of the architectures [Resnet-18](https://arxiv.org/pdf/1512.03385.pdf) and [VGG-16](https://arxiv.org/pdf/1409.1556.pdf).

For this analysis will use training/validation/holdout results and gradient-based visualization techniques from models trained on a subset of classes (between 10 and 50) from the Google Landmark Recognition Challenge Dataset.

![Cover Image | 1000x700](https://github.com/AdrianIordache/DeepLearning-In-Pytorch/blob/master/Google-Landmark-Recognition-Challenge/result-images/Presentation/Presentation-Image.png)

 
 
## Vanilla Backpropagation Results:

Original                  | Resnet-18                 |  VGG-16
:------------------------:|:-------------------------:|:-------------------------:
![Class](https://github.com/AdrianIordache/DeepLearning-In-Pytorch/blob/master/Google-Landmark-Recognition-Challenge/visualizations/input_images/Class-3.jpg)                | ![Resnet-18](https://github.com/AdrianIordache/DeepLearning-In-Pytorch/blob/master/Google-Landmark-Recognition-Challenge/result-images/Class-3/VanillaBackprop_Model_resnet18_Class_3_Normalize_True_Saliency_False.jpg)            |  ![VGG-16](https://github.com/AdrianIordache/DeepLearning-In-Pytorch/blob/master/Google-Landmark-Recognition-Challenge/result-images/Class-3/VanillaBackprop_Model_vgg16_Class_3_Normalize_True_Saliency_False.jpg)
![Class](https://github.com/AdrianIordache/DeepLearning-In-Pytorch/blob/master/Google-Landmark-Recognition-Challenge/visualizations/input_images/Class-0.jpg)                | ![Resnet-18](https://github.com/AdrianIordache/DeepLearning-In-Pytorch/blob/master/Google-Landmark-Recognition-Challenge/result-images/Class-0/VanillaBackprop_Model_resnet18_Class_0_Normalize_True_Saliency_True.jpg)            |  ![VGG-16](https://github.com/AdrianIordache/DeepLearning-In-Pytorch/blob/master/Google-Landmark-Recognition-Challenge/result-images/Class-0/VanillaBackprop_Model_vgg16_Class_0_Normalize_True_Saliency_True.jpg)
