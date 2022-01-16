from skimage.filters import threshold_otsu, sobel
from skimage import feature
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from WebcamVideoStreamFPS import webcamVideoStreamFPS

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter



def MakeKernel( x ):
    total = cv2.circle(np.zeros((x,x), np.uint8),(int(x/2),int(x/2)),int(x/2) ,1,-1);
    return total

dim=(600,800)


path_img = sys.argv[1]
start = time.time()
img  = cv2.imread(path_img)
img  = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Threshold
thresh = threshold_otsu(gray)
ret,imgThresh = cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)

#OpenClose
img_erosion = cv2.erode(imgThresh, MakeKernel(10), iterations=1)
img_dilation = cv2.dilate(img_erosion, MakeKernel(15), iterations=1)


#Mask
res = cv2.bitwise_and(gray,gray,mask = img_dilation)

#Threshold mask
ret,imgThresh = cv2.threshold(res,thresh,255,cv2.THRESH_BINARY)

#OpenClose
img_erosion = cv2.erode(imgThresh, MakeKernel(5), iterations=1)
img_dilation = cv2.dilate(img_erosion, MakeKernel(10), iterations=1)


#Ellipse
res = cv2.bitwise_and(gray,gray,mask = img_dilation)
corners = cv2.goodFeaturesToTrack(img_dilation, 300, 0.01, 10) 
img_dilation = cv2.cvtColor(img_dilation, cv2.COLOR_GRAY2BGR)
corners = np.int0(corners)
for i in corners:
	x,y = i.ravel()
	cv2.circle(img_dilation,(x,y),3,255,-1)


result = hough_ellipse(res, accuracy=20, threshold=250,min_size=100, max_size=120)

end = time.time()
print(end - start)


cv2.imshow('thresh', img_dilation)
cv2.waitKey(0)

cv2.imshow('thresh', res)
cv2.waitKey(0)

