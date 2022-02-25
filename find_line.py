import cv2
import matplotlib.pyplot as plt
import numpy as np

frame = cv2.imread('frame.png')

HSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
H_orange = 52//2
orange_lower = np.array([H_orange-5,30,40],dtype=np.uint8)
orange_upper = np.array([H_orange+5,230,240],dtype=np.uint8)

H_white = 125//2
white_lower = np.array([H_white-10,10,110],dtype=np.uint8)
white_upper = np.array([H_white+10,30,200],dtype=np.uint8)

ROI = np.zeros(frame.shape,np.uint8)
y,x = frame.shape[:2]
pts = np.array([[0,y-100],[x,y-100],[5*x//8,int(y*0.6)],[4*x//10,int(y*0.6)]])
ROI = cv2.fillPoly(ROI,[pts],(255,255,255))
ROI_img = cv2.bitwise_and(frame,ROI)

HSV_ROI = cv2.cvtColor(ROI_img,cv2.COLOR_BGR2HSV)
thresh_orange = cv2.inRange(HSV_ROI,orange_lower,orange_upper)
thresh_white = cv2.inRange(HSV_ROI,white_lower,white_upper)
roads = cv2.add(thresh_orange,thresh_white)

# plt.imshow(ROI_img)
plt.imshow(ROI_img,cmap='gray')
plt.show()

