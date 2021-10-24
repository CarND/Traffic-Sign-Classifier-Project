# **Traffic Sign Recognition** 

## Writeup


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

[image1]: ./examples/sign_images_ids_description.png "Images of all sign classes with associated id and descriptions"
[image2]: ./examples/original_data_distribution.png "Distributon of original data by class"
[image3]: ./examples/balanced_data_distribution.png "Distributon of balanced data by class"
[image4]: ./examples/internet_sign_images_ids_descriptions.png "Internet sign images with correct class ids and descriptions"
[image5]: ./internet_signs/double_curve-21.png "Double Curve"
[image6]: ./internet_signs/speedlimit30-1.png "Speed Limit 30"
[image7]: ./internet_signs/speedlimit50-2.png "Speed Limit 50"
[image8]: ./internet_signs/speedlimit70-4.png "Speed Limit 70"
[image9]: ./internet_signs/speedlimit100-7.png "Speed Limit 100"
[image10]: ./internet_signs/yield-13.png "Yield"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 41471
* The size of the validation set is ? 5183
* The size of test set is ? 10367
* The shape of a traffic sign image is ? (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the original data is distributed by class and grouped by training, validation, and testing. Note that the split of training vs testing data is close to even (with a bias towards more training than testing) across many image sign classes.

![Distributon of original data by class][image2]

Training with the above data as is did not help us reach the 93% accuracy threshold requirement for this project.  So the data was recombined, regrouped by sign class, and within each group, randomly split to training(70%), validation(10%), and testing(20%). The hope was to have our data balanced so that our model had training data that was varied and numerous enough to perform predictions above the 93% threshold.  The bar chart chart below is of the  rebalanced data distributed by class and grouped by training, validation, and testing as before. Note that the split of training vs testing data is greatly biased towards the training data so there are many more samples in the training data relative to the testing data across many image sign classes. This rebalancing was enough to improve our model's prediction accuracy to about 94% accuracy without data aumentation, normalization, or grayscaling.

![Distributon of balanced data by class][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Originally, I did preprocessing such as normalization and grayscaling.  I also tried data augmentation for classes that had fewer samples by doing translations, flipping, and adding noise.  Those initial efforts didn't prove fruitful as I was not able to achieve the 93% accuracy.  Realizing that having good and balanced data was important, I decided to pool all the data together and split the data evenly to training, validation, and testing by class.  This was enough to get me over the hump without the preprocessing and data augmentation steps.
So, I have excluded those meaning but unnecessary steps altogether.  

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | outputs 10x10x16     							|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x66 				    |
| Flatten				| outputs 400									|
| Fully connected		| input 400 output 350        					|
| Fully connected		| input 350 output 300        					|
| Fully connected		| input 300 output 120        					|
| Fully connected		| input 120 output 84        					|
| Fully connected		| input 84 output 43        					|
| Softmax				|         									    |

 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used based on the Lenet model but deeper with more fully connected layers. The AdamOptimizer was used as other optimizers (SGD, Adagrad) seemed not to perform as well.  The loss seemed to converge quite quickly at around 5 epochs so I added an extra 2 epochs, or 7 total, to account for the possibility that some training validation results do not exceed 93% accuracy at the end of 5 epochs.  A batch size of 168 was used but higher or lower numbers didn't seem to have as great an impact on the model accuracy as the quality of the data.  The learning rate was kept at .001 since a significant digit difference in either direction resulted in reduced accuracy.  

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? .940
* test set accuracy of ? .943


If a well known architecture was chosen:
* What architecture was chosen? Lenet
* Why did you believe it would be relevant to the traffic sign application? Lenet is a proven architecture for multi-class classification and as traffic signs are also multi-class, the choice to use it was obvious.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  The convergence and stabilization of the validation data accuracy after 5 epochs and it's similar performance to the test data accuracy indicated that our model was not overfitting or underfitting.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![Double Curve][image5]
![Speed Limit 30][image6]
![Speed Limit 50][image7]
![Speed Limit 70][image8]
![Speed Limit 100][image9]
![Yield][image10] 


The images The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model's prediction was surprisingly low, at 50%, at first glance.  But, after visualizing the features that the model used to classify the image, it may be reasonable to attribute a part of the failure to the actual images themselves.  The training dataset was composed of images that were more or less framed within the borners whereas some of the downloaded images were cropped inside of the sign.  One curiosity was that the 30 km sign was mislabeled for a left turn ahead sign.  Upon analysis of the image paired with the understanding that our model detects edge features, warped nature of our 30 km sign and it's left pointing lines could have confused our classifier. Perhaps downloaded images that are closer to the processing and cropping consistency of our training dataset might yield better prediction results.

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double Curve      	| Double Curve   								| 
| Speed Limit 30     	| Left Turn Ahead 								|
| Speed Limit 50		| No passing for vehicles over 3.5 metric tons	|
| Speed Limit 70	    | Speed Limit 70					 			|
| Speed Limit 100		| Speed Limit 100      							|
| Yield			        | Keep right     							|

The model was able to correctly guess 3 of the 6 traffic signs, which gives an accuracy of 50%. This compares unfavorably to the accuracy on the test set of approximately 94%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is absolutely sure that this is a double curve sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Double curve  	                        	| 
| .0     				| Right-of-way at the next intersection			|
| .0					| Road narrows on the right 					|
| .0	      			| Pedestrians					 				|
| .0				    | Wild animals crossing	     					|

For the second image, the model is very sure that this is a speed limit sign of 100 km/h (probability of 0.91930), and the image does contain that. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.91930        		| Speed limit (100km/h)  						| 
| 0.08047     			| Speed limit (120km/h) 						|
| 0.00012				| Speed limit (80km/h)  						|
| 0.00005	      		| Ahead only					 				|
| 0.00004			    | Roundabout mandatory      							|

For the third image, the model is definitely sure that this is a turn left ahead sign(probability of 0.99986), and the image actually contains a Speed limit (30km/h). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99986        		| Turn left ahead 	        					| 
| 0.08047     			| Keep right				            		|
| 0.00012				| Yield  						                |
| 0.00005	      		| Right-of-way at the next intersection			|
| 0.00004			    | Speed limit (30km/h)  						|

For the fourth image, the model is very sure that this is a No passing for vehicles over 3.5 metric tons sign (probability of  0.92358), and the image actually contains a Speed limit (50km/h). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.92358        		| No passing for vehicles over 3.5 metric tons	| 
| 0.07053     			| Turn left ahead       	            		|
| 0.00539				| Ahead only					                |
| 0.00031	      		| Dangerous curve to the right	            	|
| 0.00018			    | Yield                 						|

For the fifth image, the model is definitely sure that this is a Speed limit (70km/h) (probability of  0.99996), and the image actually contains a Speed limit (70km/h). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99996        		| Speed limit (70km/h)                      	| 
| 0.00004    			| Speed limit (30km/h)       	            	|
| 0.00000				| ASpeed limit (20km/h)					        |
| 0.00000	      		| Speed limit (80km/h)	            	        |
| 0.00000			    | Speed limit (50km/h)                			|


For the sixth image, the model is relatively confident that this is a Keep right sign (probability of  0.99996), and the image actually contains a yield sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.82048       		| Keep right                                 	| 
| 0.16942   			| Yield       	                            	|
| 0.00261				| Priority road	            			        |
| 0.00215	      		| Ahead only	                       	        |
| 0.00161			    | Roundabout mandatory               			|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Analysis of the feature maps indicates that edges were used by the model to make classification.  There are features that the model learned in 


