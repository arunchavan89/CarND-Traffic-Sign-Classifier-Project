## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)


### Dataset and Repository

1. Download the [data set](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**
[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./examples/dataset_examples.png "All images"
[image10]: ./examples/model_architecture.png "model_architecture images"

---
### Data Set Summary & Exploration

#### 1. Summary of the data set. 

* Download the [data set](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip). This is a pickled dataset in which the images are 32x32 size and in RGB color space. It contains a training, validation and test set.
* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
![alt text][image9]

### Design and Test a Model Architecture

#### 1. Preprocessing Stage
* The training, validation and test data are first preprocessed using the following two techniques.
* Firstly, the images are converted from RGB to Grayscale colorspace as shown below
![alt text][image2]
* Secondly, the images are normalized using formula image = (image - 128)/128

#### 2. Describe the final model architecture.

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							|
| Preprocessing         		| 32x32x1 Gray image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Activation Function											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					| Activation Function												| 
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				|
| Flatten	    | Input = 5x5x16. Output = 400     									|
| Fully connected		| Input = 400 Output = 120      									|
| RELU					| Activation Function												| 
| Fully connected		| Input = 120 Output = 84      									|
| RELU					|												| Activation Function
| Fully connected				| Input = 84. Output = 43      									|

![alt text][image10]

#### 3. Description about trained model. 
* The following parameters are used to tune the performance.
* Learing rate : 0.001
* EPOCHS = 100
* BATCH_SIZE = 128
* mu = 0 (mean)
* standard deviation = 0.1 (sigma)

#### 4. Results

* validation set accuracy 
 * EPOCH 100.
 * Validation Accuracy = 0.949
* Test Accuracy for 'traffic-signs-data/test.p' = 0.92

### Test a Model on New Images

#### 1. The following five German traffic signs found on the web and their prediction is obtained by using the network designed.
Here are five German traffic signs 
![alt text][image9]

#### 2. Predictions

* Test Accuracy for five pictures of German traffic signs = 0.60
* The model predicted 5 out of 5 signs correctly, it's 100.0% accurate on the input images.



