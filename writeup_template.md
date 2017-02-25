#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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
[image4]: ./myPics/60_sign.jpg "60 sign"
[image5]: ./myPics/Construction.jpg "Construction"
[image6]: ./myPics/prio_road.jpg "Priority Road"
[image7]: ./myPics/Stop_sign.jpg "STOP"
[image8]: ./myPics/yield.jpg "Yield"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 42



####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

See the code celles under _Preprocessing_ in the notebook.

The only preprocessing I did was to shuffle the training datasetets in order to avoid the apparent locational dependencies.
I choose to go with RGB images as provided in the dataset. Gray scale would eliminate important color information in this particlular case. European traffic signs are in color which helps classify the seen images.

One random example is shown from the training dataset, with the corresponding class designation.

![alt text][image2]

I made tests with both normalized and raw image data. The normalization did not seem to give any advantages, so I decided to proceed without. (the relevant lines are commented out).


#### 2. Trainining, validation and testing


To cross validate my model, I used the validation dataset in order prevent the model from overfitting.  The sizes of the training, validation and testing datasets are as follows:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630




![alt text][image3]



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the cells under _Model Architecture_

My final model consisted of the following layers:
* VALID padding is used everywhere
* Zero vectors are used for bias to start with

| # | Layer         	|   Input | Output    | Description	        		| 
|:--|:-----------------:|:-------:|:---------:|:---------------------------| 
| 1 | Convolutional     | 32x32x3 | 28x28x6   |         |
|   | Activation        |         |           | Drop Out: keep prob: 90%,  RELU, |
|   | Max Pooling       |28x28x6  | 14x14x6   | ksize: [1, 2, 2, 1], strides :[1, 2, 2, 1] |
| 2 | Convolutional     | 14x14x6 | 10x10x16  |          |
|   | Activation        |         |           | Drop Out: keep prob: 90% RELU, |
|   | Max Pooling       |10x10x16 | 5x5x16    | ksize: [1, 2, 2, 1], strides: [1, 2, 2, 1] |
|   | Flattening        |5x5x16   | 400       | One dimensional array                      |
| 3 | Fully connected   |400      | 120       | One dimensional array                      |
|   | Activation        |         |           | RELU                          |
| 4 | Fully connected   |120      | 84       | One dimensional array                      |
|   | Activation        |         |           | Drop Out: keep prob: 70%, RELU             |
| 5 | Final FC layer    |84       | 43       | Provides the logits (43 classses)          | 
|   | softmax           |43       | 43       | Final step, result of classification          | 

								
 

#### 4. Training

The code for training the model is located under _Training_

The following parameters were used during testing and validation:
* Learing rate: 0.001
* Batch size: 128
* Number of Epochs: 20
* 



####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

 Image			        |     Prediction	  					| Right class          |
|:---------------------:|:-------------------------------------:|:----------------     |
| ![image4]      		| Speed limit (60km/h)					| Speed limit (60km/h) |
| ![image6]      		| Right-of-way ..                       | Right-of-way...      |
| ![image5]      		| ~~Speed limit (70km/h)~~				| Right of way         |
| ![image7]      		| STOP              					| STOP                 |
| ![image8]      		| Yield	                				| Yield                |



The speed limits are difficult to recognize as 50 and 60 are graphically prette close to each other on this 32x32 scale.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 