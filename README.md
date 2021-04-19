# Face Emotion and Gender Recognition using Python

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


![im1](https://user-images.githubusercontent.com/82361158/115291242-21c78680-a172-11eb-88b3-f39b217356ea.JPG)

# Live face recognition using the K-nearest neighbour (K-NN) algorithm

 We first recorded the faces who desire to use this live face recognition application using webcam and for each face, a python file is generated.
 We have used OpenCV to instantiate a camera object to capture images, applied the haar cascade which is a xml file for encoding and is used to detect faces in the current frame
Extracted the face component from the image.We have converted the captured data to a numpy format and then saved the data as a numpy matrix in an encoded format.
 For the recognition part,We first detected the face,flatten it into a linear array and then pass to KNN function along with all the data.
As a result of the simulation,a rectangular frame is generated with the name of the person labelled accurately.


![im4](https://user-images.githubusercontent.com/82361158/115291834-d792d500-a172-11eb-8c46-d9a065e209f4.JPG)
# Live emotion recognition using Convolutional Neural Network(CNN)

Under this an attempt is made to to recognize user emotion using a convolutional neural network (CNN).
The neural net can recognize 7 emotions with relatively high accuracy: (1) Anger, (2) Disgust, (3) Fear, (4) Happy, (5) Sad, (6) Surprise and (7) Neutral.
We have used dependencies like OpenCV, TensorFlow and Keras to implement this.
The algorithm is able to detect different expressions of captured images and display the expression of an individual. 
However sometimes,the expressions mislead and the program at times do not accurately detect the expression
which is a scope of improvement.
We have implemented the code under Spyder IDE

![im2](https://user-images.githubusercontent.com/82361158/115291520-7cf97900-a172-11eb-9f3c-01641c61bbd9.JPG)

# Face Recognition and gender determination by training the Convolutional Neural Network(CNN) 

We have augmented the dataset to get higher accuracy rate during training of neural network.
We have divided the data set into 85% training set and 15% test set and fed them into a CNN network.
Graph of training accuracy Vs number of epochs and data loss Vs number of epochs generated to represent the accuracy details.
We can observe that as the number of iterations (epochs) is increased, we can see that the accuracy on both training and test set improves. 
The loss goes down as epochs are increased.
Out of 753 images, 740 of them were correctly identified along with their gender.
The code has been successfully implemented in Jupyter Notebook achieving a significant accuracy rate of 95.3%

![im3](https://user-images.githubusercontent.com/82361158/115292081-280a3280-a173-11eb-970a-8a703ac1a944.JPG)


