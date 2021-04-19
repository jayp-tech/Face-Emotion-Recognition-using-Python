# Face and Emotion-Recognition-using-Python

![image](https://user-images.githubusercontent.com/82361158/115289694-7d911000-a170-11eb-82fb-85d0638143e3.png)

In this project we have a large database of faces of different people with different expressions.  Our aim is to identify correct face from given dataset. We are also doing Live face recognition using the K-nearest neighbour (K-NN) algorithm and Live emotion recognition using Convolutional Neural Network(CNN).

# Implemented Approaches
Fetching corresponding RGB image from given sketch using Principal Component Analysis(PCA)

Live face recognition using the K-nearest neighbour (K-NN) algorithm.

Live emotion recognition using Convolutional Neural Network(CNN)

Face Recognition and gender determination by training the Convolutional Neural Network(CNN) to achieve high accuracy in the given dataset

# RGB image transformation from sketch image using Principal Component Analysis(PCA)

Using Principal Component Analysis(PCA) approach, we found out the eigenfaces and the basis vectors, thereby extracting the feature faces from the given data set.
The test image was converted first into a grayscale image and its corresponding weights in terms of PCA were found. 

The Euclidean difference of the test image with every RGB face of a dataset was taken and the one with the minimum difference was declared as an appropriate RGB image match to the corresponding sketch image.

The code has been successfully implemented in Colab by importing libraries like pandas, numpy, matplotlib and a few others achieving an accuracy of 73%.
