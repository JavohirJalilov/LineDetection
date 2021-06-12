import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture('roads_line.mp4')

ret,frame = cap.read()
HSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
H = 200//2
line_lower = np.array([90,140,100],np.uint8)
line_upper = np.array([110,200,180],np.uint8)
ROI = np.zeros(frame.shape,np.uint8)
y,x = frame.shape[:2]
pts = np.array([[0,y],[x,y],[5*x//9,int(y*0.6)],[4*x//9,int(y*0.6)]])
ROI = cv2.fillPoly(ROI,[pts],(255,255,255))
ROI_img = cv2.bitwise_and(frame,ROI)
thresh = cv2.inRange(HSV,line_lower,line_upper)

cv2.imshow('image',ROI_img)
cv2.waitKey(0)

cv2.release()
cv2.destroyAllWindows()