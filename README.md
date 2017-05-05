# Behavioral Cloning 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image]: ./IMG/center.jpg "160x320 image"
[cropped]: ./IMG/center_lines.jpg "cropping areas"
[center]: ./IMG/center_angle.jpg "cropping areas"
[left]: ./IMG/left_angle.jpg "cropping areas"
[right]: ./IMG/right_angle.jpg "cropping areas"
[NVIDIA]: ./IMG/nvidia.png "NVIDIA model"
[summary]: ./IMG/summary.png "Keras summary"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

To create this model, I used Keras, which in this case relies on the Tensorflow library as its backend. Coding in Keras is more concise and intuitive than in Tensorflow.

I first wanted to see if a simple convolution neural network would be efficient enough to drive the car in the simulator. For this I relied on the LeNet5 model, which was successful at categorizing 43 traffic signs categories in my previous project. 

Surprisingly, LeNet5 proved adequate at driving the car around the first track in the simulator. However, I realized looking at the model summary in Keras that my final LeNet5-inspired model contained about 1.9 millions trained parameters, which made me question whether there was an overfitting problem.

I thus decided to see if a more complicated model, such as the one designed by NVIDIA for self-driving cars, might prove better for the task. While the NVIDIA model has a more complex architecture, it has only 252 thousands trained parameters.

To recognize track features my model uses 4 convolution layers, a pooling layer, a flatten layer and three fully connected layers. 

The many convolution layers keep on getting deeper, in the hope of recognizing higher-level features on each image.

The model is fed pre-processed images (i.e. resized and cropped) but also manipulates each picture by normalizing them (using a Keras lambda layer).

To include nonlinearity, multiple RELU activation layers are used throughout the model.

Numerous models were tested by running them through the simulator and ensuring that the vehicle could stay on the track. Every time a model failed, additional changes were made to the model and the data set.



#### 2. Attempts to reduce overfitting in the model

To help reduce overfitting, the model contains 1 pooling layer after the first convolution layer, and also dropout after the first fully connected layer. 

I also limited the number of epochs to train the model to 4.


#### 3. Model parameter tuning

The model uses an adam optimizer, so the learning rate was not tuned manually. The Adam optimizer tends to converge faster than Stochastic Gradient Descent because it uses momentum, yet at the later training stages it will reduce the learning rate to help find a local minimum.

#### 4. Appropriate training data

Proper training data was critical to train the model to accurately predict the steering angle of the car. I initially relied on the Udacity set, because it features proper driving. I later added a set with my own driving data around the first track. That set includes clockwise and counter-clockwise driving (the first track contains a majority of left turns, so recording counter-clockwise driving helps the model avoid being biased towards turning left). I also added examples of recovery driving at different stages of the track (e.g. from a yellow line, from a dirt side, from side of bridge, etc). 

Using a generator, I was able to use a large data set to train the model (about 33K images for the training set and 8K for the validation set).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

When looking at the different features of the first track in the simulator, I realized that not too many would need to be successfully recognized by the model in order to drive successfully. Of great importance are the features that indicate the edges of the track:
* yellow lines
* red-and-white stripes
* black walls of the bridge
* dirt areas on the side of the road

Of course, the model would need to not only recognize a feature, but also its angle compared to the track and the car position. In my previous project, the LeNet5 model, with its two convolution layers, had proven successful at categorizing 43 different traffic signs. I thought using a model architecture with more convolutional layers might succeed at recognizing high-level features from each driving picture, and thus learn to drive the car in the simulator. 

I also had to think about the pictures fed into the model. Unlike my previous project, which used 32x32 pixels, each picture had an initial size of 160x320 pixels. The data set from Udacity is almost 400 MB and my own data set is 250 MB. Since there might have been some memory issues when trying to train the model on a larger data set, I used a generator that would read 32 images at a time.

Not all the pixels in a picture are also relevant: the top pixels only show parts of the sky and horizon, whereas the bottom pixels show mostly the hood of the car. These were cropped before the rest of the picture was analyzed by the model.

I also had to think about the tool used to read images in the first place. Initially I used the cv2 libary of OpenCV. However I later realized that OpenCV by default reads images as BGR, whereas in drive.py - the file used to drive the car in the simulator - the images were being read as RGB. This created issues because a model trained on BGR data to recognize "Yellow" markings for example would not recognize those Yellow marking when being fed RGB data. So I changed the code and used matplotlib.image (which uses RGB) instead to read the pictures when training the model.

I also thought about resizing/downsizing pictures. This would have further reduced the size of the output at the end of each convolution layer, thereby speeding the training process. The NVIDIA network uses 66x200 images, so I had to resize images to fit these measurements. These prove relatively easy to do, as the cv2 library can resize a picture based on various parameters. 

* Initial image (160x320 pixels):

![image]

* Cropped image (66x320 pixels):

![cropped]

The only additional work required was to also modify the drive.py file, because I knew the pictures being captured during the automated driving would still be of size 160x320, but my model was trained on 66x200 pictures. So I added similar cv2.resize formulas to process the pictures before they were fed to the trained model.

Confident that the generator would help prevent memory issues, I used a combined data set (Udacity and my own) that comprised about 40,000 pictures. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Because my data set was failry large, I decided to allocate 20% of the total set to the validation set. When running the model I could see the utility from Keras showing me the progress of each epoch, and that the loss was decreasing with each batch processed. After only the first epoch, loss for the validation set was around 0.016 and diminished slightly over the next 3 epochs. 

To combat overfitting, I limited the number of epochs to train my model. I also used a large data set so that it would be harder for the model to "learn" the right predicion for just a specific set of images. Finally, I used dropout, and a max pooling layer to generalize the results of the first convolution layer.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is following closely the architecture from NVIDIA.
* NVIDIA model:

![NVIDIA]


During image processing steps, each image is first normalized using a Keras Lambda layer 

The resulting layer is then fed into 5 progressively deeper convolutional layers (the first three use a 5x5 filter, whereas the last two use a 3x3 filter).

The result of the last convolutional layer is flattened, then fed into a 100-neuron fully connected layer, then a 50-neuron fully connected layer, then a 10-neuron fully connected layer.

The output layer contains only one neuron, which ouputs a number that will be used to determine the steering angle for the car.

Here is a summary of the network created by Keras:

![summary]


#### 3. Creation of the Training Set & Training Process

Capturing good driving behavior was difficult at first, as using the keyboard keys to drive the car resulted in jittery moves and made it difficult to keep the car in the center of the road, which is essential if we want the model to learn proper driving. I thus decided to use the Udacity data set because it contains examples of good driving.

I later learned to use the mouse to steer the car. This resulted in much smoother driving, as the mouse enabled me to keep steering angles constant over multiple frames. I then decided to do multiple examples of recovery driving, where I started the recording from a side of the road, steering the car back to the middle. 

Finally, I decided to use the left and right "cameras" pictures. Indeed, when recording a run, three pictures where recorded at a time: 
* one from a center camera
* one from a left camera
* one from a right camera

While it was easy to write the code to use all three sets of pictures to train the model, a more difficult part was to decide what angle to use for the left and right pictures. Indeed, the only steering angle provided was for the center picture, so a correction had to be given to that initial angle to give the model the correct angle for the right and left pictures. 

Analyzing some images, I determined the left and right camera to be at a roughly 20 degrees angle from the center view. Since the maximum steering angle of the car is +/- 25 degrees, then 20 degrees would result in a steering coefficient output from the model of +/- 0.8. However, I assumed assigning such a high correction coefficient would result in jittery driving, so I tried instead correction coefficients between 0.1 and 0.3, and settled for 0.2.
* center camera image (red line represents reference angle)

![center]

* left camera image (20 degrees deviation from reference angle)

![left]

* right camera image

![right]


Once I had enough training data that included various driving (clockwise and counter clockwise) and recovery situations, I shuffled the data before feeding it to my model.

Because of the size of my data set, I limited the number of epochs to 4, both to prevent overfitting but also because training the model took time (about 40 minutes on a g2.2xlarge instance on Amazon Web Services). I also decided to proceed cautiously when making change to my model, choosing to only update one parameter at a time (e.g. changing left/right correction coefficient from 0.1 to 0.2), because I wanted to better understand the impact of each parameter on the model. In the end, the model designed was able to drive around the first track.
