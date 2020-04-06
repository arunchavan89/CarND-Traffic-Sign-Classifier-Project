# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


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
[image11]: ./examples/german_dataset_examples.png "All images"

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
![alt text][image1]
### Design and Test a Model Architecture

#### 1. Preprocessing Stage
* The training, validation and test data are first preprocessed using the following two techniques.
* Firstly, the images are converted from RGB to Grayscale colorspace as shown below
![alt text][image2]
* Secondly, the images are normalized using formula image = (image - 128)/128

#### 2. Describe the final model architecture.
![alt text][image10]

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

A LeNet-5 is a classic Convolutional Network architecture with state-of-the-art results on the GTSRB traffic sign dataset implemented in this paper [Traffic Sign Recognition with Multi-Scale Convolutional Networks](https://scholar.google.es/scholar?q=traffic+sign+recognition+with+multi-scale+convolutional+networks&hl=en&as_sdt=0&as_vis=1&oi=scholart).
In traditional ConvNets approach, the output of the last stage is fed to a classifier. However this architecture uses the outputs of all the stages and are fed to the classifier. This helps the classifier to use, not only high-level features, which tend to be global, invariant, but also pooled lowlevel features, which tend to be more local, less invariant.

In order to achieve the validation Accuracy greater than 0.93, epochs and batch size are key parameters. By trial and error method, these parameters have set to EPOCHS = 100 and BATCH_SIZE = 128.

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
 * Validation Accuracy = 0.931
* Test Accuracy for 'traffic-signs-data/test.p' = 0.93

### Test a Model on New Images

#### 1. The following five German traffic signs found on the web and their prediction is obtained by using the network designed.
Here are five German traffic signs 
![alt text][image11]

* From the exploratory visualization of the data set, it is clear that the following signs have only few training data.
    * Speed limit (20km/hr)
    * Dangerous curve to the left
    * Go straight or left
    * End of no passing 
    * End of no passing by vehicle over 3.5 metric tons.
* The above signs have the lowest number of training datasets. This can be a major issue while predicting.
* Data augmentation can help to create more data. But it is not the best idea, as some classes remain significantly less represented than the others.
* Training a model with such an extended dataset may make it biased towards predicting overrepresented classes. 
#### 2. Predictions

* Test Accuracy for five pictures of German traffic signs = 0.80
* The model predicted 4 out of 5 signs correctly, it's 80.0% accurate on the input images.

#### 3. The top five softmax probabilities of the predictions on the captured images are outputted
* Model's softmax probabilities =  
    * [ 51.195377 ,   9.223745 ,   8.406162 ,   6.123324 ,   4.549276 ],
    * [ 52.432545 ,  19.051453 ,   4.9050813,  -4.416319 ,  -7.427154 ],
    * [ 28.191183 ,  18.278221 ,  11.604327 ,  10.93316  ,   5.842381 ],
    * [ 53.0951   ,  37.722107 ,  33.656933 ,  33.580917 ,  22.857698 ],
    * [129.80821  ,  73.44286  ,  37.12655  ,  32.33492  ,   6.8059974]],
* Indices array =
    * [14, 33, 38,  2, 40],
    * [12, 40,  9, 11,  3],
    * [40,  2, 11, 31, 21],
    * [ 7, 40,  8,  5,  4],
    * [ 4,  1,  0,  5, 40]

The above top 5 softmax probabilities shows the certainty of the model's predictions for each of the inut image. The designed model failed to detect the sign 'speed limit 30' because this image may contain less sharp edges after rescaling it to 32 x 32 size. The model is predicting this sign as 'Roundabout Mandatory' may be because of presence of sharp circular edges.  


