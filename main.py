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
HSV_ROI = cv2.cvtColor(ROI_img,cv2.COLOR_BGR2HSV)

thresh = cv2.inRange(HSV_ROI,line_lower,line_upper)
# adge = cv2.Canny(thresh,300,400)
linesP = cv2.HoughLinesP(thresh,1,np.pi/180,threshold=50,minLineLength=40,maxLineGap=100)
for line in linesP:
    for (x1,y1,x2,y2) in line:
        cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)

print(len(linesP))
cv2.imshow('image',frame)
cv2.waitKey(0)

cv2.release()
cv2.destroyAllWindows()