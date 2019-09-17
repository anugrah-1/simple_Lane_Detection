#usual imports
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os


#returns grayscale image
def grayscale(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#returns gaussina blur/smoothed image
def gaussian_blur(img, kernal_size):
	return cv2.GaussianBlur(img, (kernal_size, kernal_size), 0)

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
def region_of_intrest(img, vertices):
	#creating a blank mask
	mask = np.zeros_like(img)
	#filling the mask
	ignore_mask_color = 255
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
	return lines, line_img

#returns final image
def weighted_img(img, init_img, a=0.8, b=1, c=0):
	return cv2.addWeighted(init_img, a, img, b, c)

#to be able to trace full line we need to seperate left and right lane line
def seperate_lines(lines, img):
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


#reading all test images from the folder "test_images/"
test_img_dir = "test_images/"
orignal_img_names = os.listdir(test_img_dir)
orignal_img_names = list(map(lambda name : test_img_dir + name, orignal_img_names))
# orignal_images = list(map(lambda image : mping.imread(image), orignal_img_names))
print(orignal_img_names)

for image_name in orignal_img_names :
	#reading an image
	image = mpimg.imread(image_name)

	#printing image stats
	print('This image is of type:', type(image), 'and the dimensions are:', image.shape)

	#convert image to grayscale
	gray = grayscale(image)

	#appling gaussian blur on the grayscale image(smoothing)
	blur_img = gaussian_blur(gray, 5)

	#appling canny edge detection
	edges = canny_edge(blur_img, 50, 150)

	#extracting region of intrest
	vertices = get_vertices_for_img(blur_img)
	masked_image = region_of_intrest(edges, vertices)

	#appling hough transform
	lines, line_img = hough_lines(masked_image, 2, np.pi/180, 15, 40, 20)

	#combining hough transformed image with the orignal image
	#result_img = weighted_img(lines, image)

	#seperate left and right lanes
	seperated_lanes = seperate_lines(lines, image)
	result_img = trace_both_lane_lines(image, vertices, seperated_lanes[0], seperated_lanes[1])

	#saving the lane lined image
	dirname = "result_images/"
	cv2.imwrite(os.path.join(dirname, image_name.split('/')[-1]), result_img)
	# plt.imshow(result_img)
	# plt.show()