# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./figures/hist_train.png "Training Dataset"
[image2]: ./figures/hist_valid.png "Validation Dataset"
[image3]: ./figures/hist_test.png "Testing Dataset"
[image4]: ./figures/classes.png "Unique Classes"
[image5]: ./figures/no_aug.png "Original Images"
[image6]: ./figures/with_aug.png "Augmented Images"
[image7]: ./figures/loss_plot.png "Loss & Accuracy Plot"
[image8]: ./data/eval/1.jpg "Speed limit (30km/h)"
[image9]: ./data/eval/3.jpg "Speed limit (60km/h) "
[image10]: ./data/eval/11.jpg "Right-of-way at the next intersection"
[image11]: ./data/eval/12.jpg "Priority road"
[image12]: ./data/eval/13.jpg "Yield"
[image13]: ./data/eval/14.jpg "Stop"
[image14]: ./data/eval/23.jpg "Slippery road"
[image15]: ./data/eval/33.jpg "Wild animals crossing"
[image16]: ./data/eval/35.jpg "Ahead only"
[image17]: ./data/eval/38.jpg "Keep right"
[image18]: ./data/eval/40.jpg "Roundabout mandatory"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/uniquetrij/CarND-P2-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I created a python class named `dataset` to encapsulate the following functionalities:
1. create a `dataset` from `numpy` arrays `X` and `y` representing the feature-vectors and the class-labels respectively. 
2. create a `dataset` from an existing `pickle` file.
3. `describe` the summary of the `dataset` using `python` and `numpy` methods.
4. chart a histogram representing the frequency of examples per unique class-label in the `dataset`.
5. chart one example from each unique class-label in the `dataset`.

This facilitates me to summarize/visualize the train, validation and test datasets with ease without rewriting much code.

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Using the `describe` function of `dataset` class  I obtained the following statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32 ✖ 32 ✖ 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is spread across various class-labels. The x-axis represents the class-labels, while the y-axis represents the number of examples in that class.

1. Training Dataset\
![alt text][image1]

2. Validation Dataset\
![alt text][image2]

3. Testing Dataset\
![alt text][image3]

Following is a chart displaying one example from each class-label.
![alt text][image4]

### Design and Test a Model Architecture

I created another class named `preprocess` that contains the following preprocessing methods to preprocess a `dataset`
1. Shuffle the `dataset`.
2. Augment the `dataset`. I can introduce new images in the `dataset` by translating and/or rotating the existing images by some fraction. 
3. Convert all images in the `dataset` to gray-scale.
4. Normalize all images in the `dataset` to have pixel values in range [-1,1].

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

There are two major reasons why I decided to generate additional data.
Firstly, the number of examples across the classes in the training set varies to a great extent. The minimum number of examples in any class is 180 while the maximum goes upto 2010. Hence for better performance, more examples are needed for classes with lesser examples.
Secondly, the images in the real world may have a different orientation angle and location in the image frame. It may also have different lighting conditions, brightness, contrast etc. Hence such augmentation may also improve performance. 

I didn't perform gray-scaling on the images because it seemed to have either no effect or rather sometimes a negative effect on the training. I assume that it is because once the color information is lost, the network is only trying to train based on only intensity values, which is not enough for this dataset.

As a last step, I normalized the image data because network performance improves if the data domain space is normalized.

I tried multiple permutations of preprocessing on the datasets and I found the following combination to be optimal w.r.t obtaining adequate accuracy in feasible time:
1. Cleanup some examples from classes that has too many of them. I decided to have at-most 1500 examples per class.
2. Augment the examples from all classes such that each class now have 2000 examples (real + augmented). Augmentation of the examples comprises of rotation of at most +/- 7.5 degrees and/or translation of at most +/- 7.5% of its length, both in horizontal and vertical directions. I didn't include lighting conditions for augmentation because I found that the dataset already contained images with varied lighting conditions.
3. Normalize all images between [-1, 1].

Below is a comparison between one set of original examples and their augmented versions from each class-labels.
* Original Images\
![alt text][image5]
* Augmented Versions\
![alt text][image6]

Note that image augmentation is performed only on the training dataset and not on validation or test datasets. The final training dataset contained 2000 examples for each of the 43 classes.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| Normalized 32x32x3 RGB image      			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Flatten               | outputs 400                                   |
| Fully connected		| outputs 120  									|
| RELU                  |                                               |
| Fully connected		| outputs 84  									|
| RELU                  |                                               |
| Fully connected		| outputs 43  									|
| Softmax				| outputs 43                                    |
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 

1. Adam Optimizer with a learning rate 0.001
2. Reduce-Mean loss over Softmax Cross Entropy with logits 
3. Max 100 epochs
4. Batch size of 128 
5. Stop training once validation accuracy reaches 96%

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I chose the LeNet Architecture for the CNN model with slight modification to accommodate the input and output feature/data shape for the traffic dataset. The LeNet architecture was first introduced for OCR (Optical Character Recognition). My task here was to detect traffic signs which is, to a large extent, similar to that of OCR. Hence I believed this architecture would be a good choice.

My final model results after 17 Epochs were:
* training set accuracy of 0.984
* validation set accuracy of 0.962 
* test set accuracy of 0.942

The figure below shows the loss and accuracy plot
![alt text][image7]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

For the model evaluation, I used the folloeing eleven German traffic signs that I found on the web:\
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]


Initially, the first two images were difficult to classify as they are very similar to other street signs in the speed-limit category. I observed that, each time I train the model, the prediction done my the trained model varied. I performed at least 100 trainings, among which the missclassification of these two classes were about 50%. Some classes were rarely misclassified. Others classes were misclassified 5-10% of the times. Also, tweeking with the augmentation parameters(amount of translation, rotation, number of augmentation), the prediction accuracy changed. Luckily on the final model, all the evaluation images were predicted correctly (so far the best trained model, as it seems).


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:


|              Actual Label             |            Predicted Label            | Probability Score |
|:-------------------------------------:|:-------------------------------------:|:------------------|
|                 Yield                 |                 Yield                 |      1.00000      |
|             Priority road             |             Priority road             |      1.00000      |
|            Turn right ahead           |            Turn right ahead           |      1.00000      |
|                  Stop                 |                  Stop                 |      1.00000      |
|               Keep right              |               Keep right              |      1.00000      |
|          Speed limit (30km/h)         |          Speed limit (30km/h)         |      0.99991      |
|          Speed limit (60km/h)         |          Speed limit (60km/h)         |      1.00000      |
|          Roundabout mandatory         |          Roundabout mandatory         |      1.00000      |
|               Ahead only              |               Ahead only              |      1.00000      |
|             Slippery road             |             Slippery road             |      0.78150      |
| Right-of-way at the next intersection | Right-of-way at the next intersection |      1.00000      |


The model was able to correctly guess all 11 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 10th cell of the Ipython notebook.

Following are the list of top 5 predictions done by the model over the eleven evaluation traffic sign images.


## 1. Actual Class:  Stop
|   Prediction  | Probability |
|---------------|-------------|
|      Stop     |   1.00000   |
|    No entry   |   0.00000   |
| Priority road |   0.00000   |
|     Yield     |   0.00000   |
|  No vehicles  |   0.00000   |


## 2. Actual Class:  Priority road
|              Prediction             | Probability |
|-------------------------------------|-------------|
|            Priority road            |   1.00000   |
|           Traffic signals           |   0.00000   |
|                 Stop                |   0.00000   |
|                Yield                |   0.00000   |
| End of all speed and passing limits |   0.00000   |


## 3. Actual Class:  Ahead only
|      Prediction      | Probability |
|----------------------|-------------|
|      Ahead only      |   1.00000   |
|   Turn left ahead    |   0.00000   |
|   Turn right ahead   |   0.00000   |
| Go straight or left  |   0.00000   |
| Go straight or right |   0.00000   |


## 4. Actual Class:  Speed limit (60km/h)
|      Prediction      | Probability |
|----------------------|-------------|
| Speed limit (60km/h) |   1.00000   |
| Speed limit (50km/h) |   0.00000   |
|     No vehicles      |   0.00000   |
| Speed limit (80km/h) |   0.00000   |
|      No passing      |   0.00000   |


## 5. Actual Class:  Right-of-way at the next intersection
|               Prediction              | Probability |
|---------------------------------------|-------------|
| Right-of-way at the next intersection |   1.00000   |
|           Beware of ice/snow          |   0.00000   |
|              Double curve             |   0.00000   |
|              Pedestrians              |   0.00000   |
|      Dangerous curve to the right     |   0.00000   |


## 6. Actual Class:  Roundabout mandatory
|      Prediction      | Probability |
|----------------------|-------------|
| Roundabout mandatory |   1.00000   |
|   Turn left ahead    |   0.00000   |
|      Keep right      |   0.00000   |
| Go straight or left  |   0.00000   |
| Go straight or right |   0.00000   |


## 7. Actual Class:  Speed limit (30km/h)
|      Prediction      | Probability |
|----------------------|-------------|
| Speed limit (30km/h) |   0.99991   |
| Speed limit (70km/h) |   0.00009   |
| Speed limit (20km/h) |   0.00000   |
| Speed limit (50km/h) |   0.00000   |
| Speed limit (80km/h) |   0.00000   |


## 8. Actual Class:  Slippery road
|       Prediction      | Probability |
|-----------------------|-------------|
|     Slippery road     |   0.78150   |
|   Bicycles crossing   |   0.21850   |
|   Children crossing   |   0.00000   |
| Wild animals crossing |   0.00000   |
|       Bumpy road      |   0.00000   |


## 9. Actual Class:  Keep right
|      Prediction      | Probability |
|----------------------|-------------|
|      Keep right      |   1.00000   |
|   Turn left ahead    |   0.00000   |
| Go straight or right |   0.00000   |
| Roundabout mandatory |   0.00000   |
|      Keep left       |   0.00000   |


## 10. Actual Class:  Yield
|                     Prediction                     | Probability |
|----------------------------------------------------|-------------|
|                       Yield                        |   1.00000   |
| End of no passing by vehicles over 3.5 metric tons |   0.00000   |
|    No passing for vehicles over 3.5 metric tons    |   0.00000   |
|                  General caution                   |   0.00000   |
|                      No entry                      |   0.00000   |


## 11. Actual Class:  Turn right ahead
|      Prediction      | Probability |
|----------------------|-------------|
|   Turn right ahead   |   1.00000   |
| Roundabout mandatory |   0.00000   |
|      Ahead only      |   0.00000   |
|   Turn left ahead    |   0.00000   |
|      Keep left       |   0.00000   |

It is amazing to see how well the model was able to predict the evaluation images. In all of the cases, it predicted the correct label with the maximum probability. In fact, in 10 out of 11 cases it predicted the correct class label with probability score of 1 (rounded off to 5 decimal places). The model seems to be slightly confused between evaluation of `Slippery road` (78%) for `Bicycles crossing` (22%). Observing the two signs, we can see that the signs are very similar, and hence this is an expected behavior of the model.

### Concluding Remarks
1. Accuracy of the model may change if we train it again from scratch. Thus the results obtained here is only specific the model of my final training instance. It may not give the same result on another model instance even if the same algorithm is followed for training.

2. The major reason behind the above behavior is the randomization in image augmentation, shuffling and weights initialization for the network.

3. Google Colab (https://colab.research.google.com/) is a great environment for model training, that made it very easy for me to work with this project. The entire training was performed on a GPU instance provided by Google Colab. The first cell of the Ipython notebook contains code for automatic resolution of dependencies (dataset, libraries etc ) and for the upload of eval images from local system. The last cell contains code to download the trained model into the local system. These code cells must be commented while running it on any other environment and the dependencies must be taken care of manually.

4. The eval images must have the same name as their class labels. If there be multiple images belonging to same class, the image name must be prefixed by the class label followed by a dot (.). Eg. if the image name is `"xyz.jpg"` belonging the class label 25, it may be named simply `"25.jpg"` or `"25.xyz.jpg"`.

5. The following optional "Visualizing the Neural Network" part was not attempted due to time constraints. I'm planning to complete that in the future.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


