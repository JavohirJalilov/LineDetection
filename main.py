import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture('roads_line.mp4')

while True:
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

    rightSlope = []
    leftSlope = []
    rightIntercept = []
    leftIntercept = []

    for line in linesP:
        for (x1,y1,x2,y2) in line:
            slope = (y2-y1)/(x2-x1)
            if slope > 0.2:
                if x1 > 300:
                    y_intercept = y2 - slope*x2
                    rightSlope.append(slope)
                    rightIntercept.append(y_intercept)
                else:
                    None
            elif slope < -0.2:
                if x1 < 400:
                    y_intercept = y2 - slope*x2
                    leftSlope.append(slope)
                    leftIntercept.append(y_intercept)

    leftavgSlope = np.mean(leftSlope)
    leftavgIntercept = np.mean(leftIntercept)
    rightavgSlope = np.mean(rightSlope)
    rightavgIntercept = np.mean(rightIntercept)

    left_line_x1 = int((0.65*y - leftavgIntercept)/leftavgSlope)
    left_line_x2 = int((y - leftavgIntercept)/leftavgSlope)

    right_line_x1 = int((0.65*y - rightavgIntercept)/rightavgSlope)
    right_line_x2 = int((y - rightavgIntercept)/rightavgSlope)

    pts = np.array([[left_line_x1,int(0.65*y)],[left_line_x2,y],[right_line_x2,y],[right_line_x1,int(0.65*y)]])

    cv2.fillPoly(frame,[pts],(255,0,0))
    cv2.line(frame,(left_line_x1,int(0.65*y)),(left_line_x2,y),(0,0,255),3)
    cv2.line(frame,(right_line_x1,int(0.65*y)),(right_line_x2,y),(0,255,0),3)

    cv2.imshow('image',frame)
    if cv2.waitKey(1) == 27:
        break

cv2.release()
cv2.destroyAllWindows()