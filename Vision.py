from skimage.filters import threshold_otsu, sobel
from skimage import feature
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from WebcamVideoStreamFPS import webcamVideoStreamFPS
from imutils.video import FPS

def MakeKernel( x ):
    total = cv2.circle(np.zeros((x,x), np.uint8),(int(x/2),int(x/2)),int(x/2) ,1,-1);
    return total

dim=(600,800)
offset=15


path_img = sys.argv[1]
start = time.time()
img  = cv2.imread(path_img)
img  = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
"""gaussian_noise = img.copy()
gaussian_noise = cv2.randn(gaussian_noise, 0, 300)
gray = cv2.cvtColor(img+gaussian_noise,cv2.COLOR_BGR2GRAY)"""

#Threshold
thresh = threshold_otsu(gray)
ret,imgThresh = cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)

#OpenClose
img_erosion = cv2.erode(imgThresh, MakeKernel(10), iterations=1)
img_dilation = cv2.dilate(img_erosion, MakeKernel(15), iterations=1)


#Mask
res = cv2.bitwise_and(gray,gray,mask = img_dilation)

#Threshold mask
thresh = threshold_otsu(res)
ret,imgThresh = cv2.threshold(res,thresh,255,cv2.THRESH_BINARY)

#OpenClose
img_erosion = cv2.erode(imgThresh, MakeKernel(5), iterations=1)
img_dilation = cv2.dilate(img_erosion, MakeKernel(10), iterations=1)


#Circles
res = cv2.bitwise_and(gray,gray,mask = img_dilation)
cimg_dilation = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(res,cv2.HOUGH_GRADIENT,1,800,param1=50,param2=30,minRadius=50,maxRadius=100)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
	# draw the outer circle
	cv2.circle(cimg_dilation,(i[0],i[1]),i[2],(0,255,0),2)
	# draw the center of the circle
	cv2.circle(cimg_dilation,(i[0],i[1]),2,(0,0,255),3)

#Line to cut
mainH = 0
for i in circles[0,:]:
	if mainH < i[1]+i[2]:
		mainH = i[1]+i[2]
if mainH==0:
	mainH=dim[1]
else:
	mainH=mainH+offset

cv2.line(cimg_dilation,(0,mainH),(dim[0],mainH),(255,0,0),2)

end = time.time()
print(end - start)

cv2.imshow('thresh', cimg_dilation)
cv2.waitKey(0)
