# Road Lane Line Detection from images and videos
Lane detetcion is a very common task performed by every human while driving. This is important to ensure there vehicles are within lane, so to make sure traffic is smooth and avoid collision with other vehicles.

Similarly, It is important in autonomous vehicles to detect lane lines.

The goal of this project is to create a pipeline to detect lane lines on the road.

## Test Data
Udacity provided test images and videos. Below is an sample image.
![Screenshot](https://github.com/anugrah-1/simple_Lane_Detection/blob/master/test_images/solidWhiteCurve.jpg)

## The Pipeline
Here, we will be going to define each and every step required in our pipeline, which will going to detect lane line.
* Converting image to grayscale
* Apply Gaussian Blur to Smoothen edges
* Apply Canny edge detetcion
* Trace Region of Interest and remove all other lines detected in previous step
* Perform Hough Transform and find lines within region of interest
* Interpolate line gradients to create two smooth lines

## Convert To Grayscale
Converting from color to grayscale image results in easier manipulations.
Here is an example of what we will get as an output.
![Screenshot](https://github.com/anugrah-1/simple_Lane_Detection/blob/master/test_images/solidWhiteCurve.jpg)
