#usual imports
from scipy import stats
from collections import deque
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math

MAXIMUM_SLOPE_DIFF = 0.1
MAXIMUM_INTERCEPT_DIFF = 50.0

rho = 1
# 1 degree
theta = (np.pi/180) * 1
threshold = 15
min_line_length = 20
max_line_gap = 10

#returns grayscale image
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#returns gaussina blur/smoothed image
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

#return canny edge detected image
def canny_edge(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

#find vertices for image
def get_vertices_for_img(img):
    img_shape = img.shape
    height = img_shape[0]
    width = img_shape[1]

    vert = None
    
    if (width, height) == (960, 540):
        region_bottom_left = (130 ,img_shape[0] - 1)
        region_top_left = (410, 330)
        region_top_right = (650, 350)
        region_bottom_right = (img_shape[1] - 30,img_shape[0] - 1)
        vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
    else:
        region_bottom_left = (200 , 680)
        region_top_left = (600, 450)
        region_top_right = (750, 450)
        region_bottom_right = (1100, 650)
        vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)

    return vert

#returns masked image with only intersting region
def region_of_interest(img):
    #creating a blank mask
    mask = np.zeros_like(img)
    #filling the mask
    ignore_mask_color = 255
    #calculate vertices
    vertices = get_vertices_for_img(img)
    #filling pixels inside the polygon defined with vertices with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returing the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#drawing lines on image using line segments
def draw_lines(img, lines, color = [255,0,0], thickness = 5):
    for line in lines:
        for x1,y1,x2,y2 in line:
            img = cv2.line(img, (x1,y1), (x2,y2), color, thickness)
    return img

#returns image with hough lines
def hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength = min_line_length, maxLineGap = max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    draw_lines(line_img, lines)
    return lines

#returns final image
def weighted_img(img, init_img, a=0.8, b=1, c=0):
    return cv2.addWeighted(init_img, a, img, b, c)

#to be able to trace full line we need to seperate left and right lane line
def separate_lines(img, lines):
    img_shape = img.shape
    middle_x = img_shape[1] / 2
    left_lane_line = []
    right_lane_line = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            dx = x2 - x1
            if dx == 0:
                #dicarding line as we can't gradient is undefined at this line
                continue
            dy = y2 - y1
            #similarly if y remain constant as x increases, discard line
            if dy == 0:
                continue
            slope = dy / dx
            #get rid of lines with smaller slope as they are likely horizontal
            epsilon = 0.1
            if abs(slope) < epsilon:
                continue

            if slope < 0 and x1 < middle_x and x2 < middle_x:
                #lane should also be in the left side of the region of intrest
                left_lane_line.append([[x1, y1, x2, y2]])
            else:
                #lane should also be in the right side of the region of intrest
                right_lane_line.append([[x1, y1, x2, y2]])
    return left_lane_line, right_lane_line

def find_lane_line_formula(lines):
    xs = []
    ys = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)
        slope, intercept, r_value, p_value, std_err = stats.linregress(xs,ys)
    #straight line represented as f(x) = Ax + B. slope is the A while intercept is B
    return (slope, intercept)

def trace_lane_line(img, lines, vertices, top_y):
    A, b = find_lane_line_formula(lines)
    img_shape = img.shape
    bottom_y = img_shape[0] - 1
    #y = Ax + b. therfore, x = (y - b) / A
    x_to_botttom_y = (bottom_y - b) / A
    top_x_to_y = (top_y - b) / A
    new_lines = [[[int(x_to_botttom_y),int(bottom_y), int(top_x_to_y), int(top_y)]]]
    return draw_lines(img, new_lines)

def trace_both_lane_lines(img, vertices, left_lane_line, right_lane_line):
    region_top_left = vertices[0][1]
    full_left_lane_line = trace_lane_line(img, left_lane_line, vertices, region_top_left[1])
    full_left_right_lane_line = trace_lane_line(full_left_lane_line, right_lane_line, vertices, region_top_left[1])
    img_with_lane_weights = cv2.addWeighted(img, 0.7, full_left_right_lane_line, 0.3, 0.0)
    return img_with_lane_weights


#lane detection using memory
#funtion to create a deque
def create_lane_line_coefficients_list(length = 10):
    return deque(maxlen = length)

#tracing lane line with coefficient
def trace_lane_line_with_coefficients(img, line_coefficients, top_y):
    A = line_coefficients[0]
    b = line_coefficients[1]
    img_shape = img.shape
    bottom_y = img_shape[0] - 1
    #y = Ax + b, therefore x = (y-b) / A
    x_to_bottom_y = (bottom_y - b) / A
    top_x_to_y = (top_y - b) / A
    new_lines = [[[int(x_to_bottom_y), int(bottom_y), int(top_x_to_y), int(top_y)]]]
    return draw_lines(img,new_lines)

def trace_both_lane_lines_with_lines_coefficients(img, left_line_coefficients, right_line_coefficients):
    vertices = get_vertices_for_img(img)
    region_top_left = vertices[0][1]

    full_left_lane_img = trace_lane_line_with_coefficients(img, left_line_coefficients, region_top_left[1])
    full_left_right_lane_img = trace_lane_line_with_coefficients(full_left_lane_img, right_line_coefficients, region_top_left[1])

    img_with_line_weight = cv2.addWeighted(img, 0.7, full_left_lane_img, 0.3, 0.0)
    return img_with_line_weight


class LaneDetectorWithMemory:
    def __init__(self):
        self.left_lane_coefficients  = create_lane_line_coefficients_list()
        self.right_lane_coefficients = create_lane_line_coefficients_list()
        
        self.previous_left_lane_coefficients = None
        self.previous_right_lane_coefficients = None
        
    
    def mean_coefficients(self, coefficients_queue, axis=0):        
        return [0, 0] if len(coefficients_queue) == 0 else np.mean(coefficients_queue, axis=axis)
    
    def determine_line_coefficients(self, stored_coefficients, current_coefficients):
        if len(stored_coefficients) == 0:
            stored_coefficients.append(current_coefficients) 
            return current_coefficients
        
        mean = self.mean_coefficients(stored_coefficients)
        abs_slope_diff = abs(current_coefficients[0] - mean[0])
        abs_intercept_diff = abs(current_coefficients[1] - mean[1])
        
        if abs_slope_diff > MAXIMUM_SLOPE_DIFF or abs_intercept_diff > MAXIMUM_INTERCEPT_DIFF:
            #print("Identified big difference in slope (", current_coefficients[0], " vs ", mean[0],
             #    ") or intercept (", current_coefficients[1], " vs ", mean[1], ")")
            
            # In this case use the mean
            return mean
        else:
            # Save our coefficients and returned a smoothened one
            stored_coefficients.append(current_coefficients)
            return self.mean_coefficients(stored_coefficients)
        

    def lane_detection_pipeline(self, img):
        grayscale_img = grayscale(img)
        gaussian_smoothed_img = gaussian_blur(grayscale_img, kernel_size=5)
        canny_img = canny_edge(gaussian_smoothed_img, 50, 150)
        segmented_img = region_of_interest(canny_img)
        hough_line = hough_lines(segmented_img, rho, theta, threshold, min_line_length, max_line_gap)

        try:
            left_lane_lines, right_lane_lines = separate_lines(img, hough_line)
            left_lane_slope, left_intercept = find_lane_line_formula(left_lane_lines)
            right_lane_slope, right_intercept = find_lane_line_formula(right_lane_lines)
            smoothed_left_lane_coefficients = self.determine_line_coefficients(self.left_lane_coefficients, [left_lane_slope, left_intercept])
            smoothed_right_lane_coefficients = self.determine_line_coefficients(self.right_lane_coefficients, [right_lane_slope, right_intercept])
            img_with_lane_lines = trace_both_lane_lines_with_lines_coefficients(img, smoothed_left_lane_coefficients, smoothed_right_lane_coefficients)
        
            return img_with_lane_lines

        except Exception as e:
            print("*** Error - will use saved coefficients ", e)
            smoothed_left_lane_coefficients = self.determine_line_coefficients(self.left_lane_coefficients, [0.0, 0.0])
            smoothed_right_lane_coefficients = self.determine_line_coefficients(self.right_lane_coefficients, [0.0, 0.0])
            img_with_lane_lines = trace_both_lane_lines_with_lines_coefficients(img, smoothed_left_lane_coefficients, smoothed_right_lane_coefficients)
        
            return img_with_lane_lines

#giving video as input

#vdeo1
white_output = "result_videos/solidWhiteRight.mp4"
detector = LaneDetectorWithMemory()
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(detector.lane_detection_pipeline)
white_clip.write_videofile(white_output, audio=False)
print("Video1 Saved!")

#video2
yellow_output = "result_videos/solidYellowLeft.mp4"
detector = LaneDetectorWithMemory()
clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
yellow_clip = clip1.fl_image(detector.lane_detection_pipeline)
yellow_clip.write_videofile(yellow_output, audio=False)
print("Video2 Saved!")

#video3 //challange video
challange_output = "result_videos/challenge.mp4"
detector = LaneDetectorWithMemory()
clip1 = VideoFileClip("test_videos/challenge.mp4")
challange_clip = clip1.fl_image(detector.lane_detection_pipeline)
challange_clip.write_videofile(challange_output, audio=False)
print("Video2 Saved!")