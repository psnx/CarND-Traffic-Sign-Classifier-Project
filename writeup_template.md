Title: CarND Project 2 Traffic Sign Recognition  
Ver: 1.0 (first submission)
Author: Tamas Panyi  
Date: 2017-02-26  
System: Ubuntu 16.04.2 LTS  
Editor: Visual Studio Code  

# **CarND Project 2 Traffic Sign Recognition** 
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./myPics/class_population.png "Population of Each class (training set)"
[image2]: ./myPics/predictions_chart.png "Predictions"
[image4]: ./myPics/60_sign.jpg "60 sign"
[image5]: ./myPics/Construction.jpg "Construction"
[image6]: ./myPics/prio_road.jpg "Priority Road"
[image7]: ./myPics/Stop_sign.jpg "STOP"
[image8]: ./myPics/yield.jpg "Yield"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

The Writeup report is herebey presented.

### Data Set Summary & Exploration

#### 1. Basic summary of the data set

The code for this step is contained in the _Exploration of Dataset_ code cell of the IPython notebook. I used the standard numpy 
panda, and tensorflow modules.
functionality to analyze the data set.

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43 (0 to 42)


#### 2. Exploratory visualization of the data set

A simple visualization of data 
_Visualisation of the dataset_

* A typical picture representing each class (first occurence in each class)
* number of samples in each class (in all three sets)

As the number of samples in each class varies a lot, I would expect that any model
based on this data set will be better at identifying the most populous classes:


|ID  | Designation | #of samples |
|:---|:------------------|:------------:|
|  2  |  Speed limit (50km/h)  |  2010 |
|  1  |  Speed limit (30km/h)  |  1980 |
|  13  |  Yield  |  1920 |
|  12  |  Priority road  |  1890 |
|  38  |  Keep right  |  1860 |
|  10  |  No passing for vehicles over 3.5 metric tons  |  1800 |
|  4  |  Speed limit (70km/h)  |  1770 |
​

![Barchart, dataset][image1]

---
### Design and Test a Model Architecture

#### 1. Pre-processing of image data

See the code cells under _Pre-process the Data Set_ in the notebook.

The only preprocessing I did was to shuffle the training data in order to avoid the apparent locational dependencies.
I choose to go with RGB images as provided in the dataset. Gray scale would eliminate important color information in this particular case. 
European traffic signs are use both colors and shapes to make signage easy to identify under poor visibility conditions.

One random example is shown from the training dataset, with the corresponding class designation.  
In the next cell I picked one representative from each class to get a look and feel of the training dataset, 
and to verify the names provided in `signnames.csv` file.

I made tests with both normalized and raw image data. The normalization did not seem to give any advantages, so I decided to proceed without it. 
(the relevant lines are commented out).  
One can see that the `signnames.csv`, and my classification correctly labels the classes. (visual images cf. corresponding class designations further down)


#### 2. Training, validation and testing

To cross validate my model, I used the validation dataset in order prevent the model from overfitting.  The sizes of the training, validation 
and testing data sets are as follows:

* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410

#### 3. Model Architecture

The code for my final model is located in the cells under _Model Architecture_

My final model consists of the following layers:

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
| 5 | Final FC layer    |84       | 43       | Provides the logits (43 classes)          | 
|   | softmax           |43       | 43       | Final step, result of classification          | 

* VALID padding is used everywhere
* Zero vectors are used for bias to start with

#### 4. Training

The code for training the model is located under _Training, Validation and Testing_

The following parameters were used during training:
* Learning rate: 0.001
* Batch size: 128
* Number of Epochs: 20

I run 20 epoch because with this learning rate and the use of dropouts made the learning
process relatively slow. Around 15 epochs the accuracy on all data sets (training, validation
and testing) flattens out.

The fact that the accuracies of the training, validation and test data sets grow together,
and that the test dataset only slightly leads the growth, allows us to conclude that the
learning is indeed real, not apparent.
This will be confirmed by the model's good performance on the random European
signs (not all of them are German, however the differences may be subtle as these signages
are greatly standardized across the old continent).

#### 5. Solution

The code for calculating the accuracy of the model is located under the _Training, Validation and Testing_ section of the Jupyter notebook.

My final model results were:
* training set accuracy of 97.6%
* validation set accuracy of 91.3%
* test set accuracy of 90.6%
* accuracy with regards to the 5 European signs: 80.0%

I chose and iterative approach to find the optimal solution.
* I started off with the LeNet model as presented in the previous lessons.
* This model was configured to be fed with gray scale images. I applied some minor adjustments to make it compatible with RGB images
* I added some dropout layers along the way, and made a few _trials and errors_ runs with random drop out ratios.
    * My goal was to avoid over fitting wile maintaining an acceptable accuracy
    * I ended up with 20 Epochs in order to cater for slow learning rate and a few dropout layers
* I kept the ReLU activation function.
* I tried to eliminate the variations from run to run by using small sigma (0.001) and zero mean. 
In this way the end result was less dependent on the starting conditions and I could tune the parameters more easily.
* In the end the difference between the accuracy on the training set and validation set is 7%, 
which provides evidence that the learning is indeed real.

--- 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

 ![image4] ![image5] ![image6] ![image7] ![image8]

The speed limits are difficult to recognize as 50 and 60 are graphically pretty close to each other on 
this small scale of 32x32 pixels.

#### 2. The model's predictions on the German traffic signs

| Image			        | Prediction	     	    | True Class           |
|:---------------------:|:--------------------------|:---------------------|
| ![image4]      		| ~~Speed limit (50km/h)~~	| Speed limit (60km/h) |
| ![image6]      		| Right-of-way ..           | Right-of-way...      |
| ![image5]      		| Construction          	| Construction         |
| ![image7]      		| STOP              		| STOP                 |
| ![image8]      		| Yield	                	| Yield                |

Four signs were recognized. The predictions  tend to be unstable after starting the training sessions from scratch 
depending on the choice of hyper parameters. This makes finding the optimal set of hyper parameters and architecture
 more difficult.

The model has as hit rate of 80% at this point. The same settings may produce sometimes a model that makes 100%
precise predictions, or sometimes as low as 60%. 


#### 3. Certainty of the model's predications on the German traffic signs

The code for making predictions on my final model is located in 
under section _Predict the Sign Type for Each Image_

Bar charts are provided for better visualization. The model is rather sure about its predictions,
even when the prediction turns out to be wrong (ie. sample 1). This is undesirable, in an 
ideal case the model would be at least somewhat uncertain when it makes a mistake.

![image2]

