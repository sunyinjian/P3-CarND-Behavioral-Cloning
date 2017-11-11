#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[error_loss]: ./image/loss.png "Error Loss"
[original_steering]: ./image/original_steering.png "Original Steering Angles Distribution"
[augmentation_steering]: ./image/augmentation_steering.png "Steering Angles Distribution After Augmentation"
[multi_cameras]: ./image/multi_cameras.png "multi_cameras"
[augment_images]: ./image/augment_images.png "augment_images"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes, depths between 24 and 64, all with batch normalization and ReLU activations (model.py lines 14-55). The shape of input image is 64X64X3 for saving training time. The data is normalized in the model using a Keras lambda layer (model.py lines 15). I aslo used batch normalization for faster learning and higher overall accuracy. I added dropout layers avoiding overfitting.

The model is based on the Nvidia's paper "End to End Learning for Self-Driving Cars", which is as follows:
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 32, 32, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
batchnormalization_1 (BatchNorma (None, 32, 32, 24)    96          convolution2d_1[0][0]            
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 32, 32, 24)    0           batchnormalization_1[0][0]       
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 16, 16, 36)    21636       activation_1[0][0]               
____________________________________________________________________________________________________
batchnormalization_2 (BatchNorma (None, 16, 16, 36)    144         convolution2d_2[0][0]            
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 16, 16, 36)    0           batchnormalization_2[0][0]       
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 8, 48)      43248       activation_2[0][0]               
____________________________________________________________________________________________________
batchnormalization_3 (BatchNorma (None, 8, 8, 48)      192         convolution2d_3[0][0]            
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 8, 8, 48)      0           batchnormalization_3[0][0]       
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 8, 8, 48)      0           activation_3[0][0]               
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 6, 64)      27712       dropout_1[0][0]                  
____________________________________________________________________________________________________
batchnormalization_4 (BatchNorma (None, 6, 6, 64)      256         convolution2d_4[0][0]            
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 6, 6, 64)      0           batchnormalization_4[0][0]       
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 4, 64)      36928       activation_4[0][0]               
____________________________________________________________________________________________________
batchnormalization_5 (BatchNorma (None, 4, 4, 64)      256         convolution2d_5[0][0]            
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 4, 4, 64)      0           batchnormalization_5[0][0]       
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 4, 4, 64)      0           activation_5[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1024)          0           dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           102500      flatten_1[0][0]                  
____________________________________________________________________________________________________
batchnormalization_6 (BatchNorma (None, 100)           400         dense_1[0][0]                    
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 100)           0           batchnormalization_6[0][0]       
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 100)           0           activation_6[0][0]               
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_3[0][0]                  
____________________________________________________________________________________________________
batchnormalization_7 (BatchNorma (None, 50)            200         dense_2[0][0]                    
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 50)            0           batchnormalization_7[0][0]       
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         activation_7[0][0]               
____________________________________________________________________________________________________
batchnormalization_8 (BatchNorma (None, 10)            40          dense_3[0][0]                    
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 10)            0           batchnormalization_8[0][0]       
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          activation_8[0][0]               
====================================================================================================
Total params: 241,003
Trainable params: 240,211
Non-trainable params: 792
____________________________________________________________________________________________________
```

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. By normalizing the data in each mini-batch, the problem referring to as 'internal covariate shift' could be largely avoided.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The learning rate was manually tuned to 0.001.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 


###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a model similar to the the Nvidia end-to-end learning networks 

My first step was to use a convolution neural network model similar to the Nvidia end-to-end learning networks. I thought this model might be appropriate because Nvidia has successfully use the model on the real world.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout layers. I used batch normalization layers for normalizing the data in the model.

Then I used keras generators to read data from the file, along with the augmentation on the fly, including random selection of left/right/center images, random flipping, shifting the image horizontally and vertivally, croping image to remove the sky and the bonnet, resizing image to 64x64 dimension to save training time.

Firstly, I only trained the model using the udacity data of track 1. The vehicle was able to drive autonomously around the track 1 without leaving the road. But the vehicle could not pass around the track 2. Then I recorded three laps on track 2 using center lane driving. Due to the limitation of my computer hardware, I only used the simulator with fastest graphics quality. Finally, I trained the model using both data of track 1 and 2. I used the function ModelCheckpoint to save the best model from the training. Here is an image of error loss:

![alt text][error_loss]

At the end of the process, the vehicle was able to drive autonomously around the track 1 and 2 without leaving the road. As for track 1, the speed could be set to 30km/h. However, the top speed for track 2 could only be set to 25km/h. In the drive.py, I adjusted the parameter "I" to 0.0002 of function SimplePIController, making the overshoot less. Otherwise, the vehicle would speed up to 30 km/h quickly and then fall off in the first turn for track 2.


####2. Creation of the Training Set & Training Process

The data quality is essential for the training. I used the following augmentation techniques in generator to extract more instances of training data.

To simulate the vehicle recovering from one side back to center, I used the multiple cameras by choosing one randomly and adjusted the steering angle for the left and right cameras. Here is an image of multiple cameras and steerings adjustment:

![alt text][multi_cameras]

To augment the data set, I also flipped images and steering angles so that this would increase the diversity of the sample data. 

The sample dataset is heavily biased towards near some steering angle values. To have a balanced histogram of steering angles, I shifted the image horizontally and add an offset to steering angle. Here are the histograms of steering angles before and after shifting:

![alt text][original_steering]
![alt text][augmentation_steering]

Then I cropped the image to remove the sky and the bonnet , resize to 64*64 dimension to save training time.
Here is an image of data augmentation :

![alt text][augment_images]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the loss change.

