import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture('sam_road.MOV')

while True:
    ret,frame = cap.read()
    cv2.imwrite('frame_dinamo.png', frame)
    HSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    H_orange = 52//2
    orange_lower = np.array([H_orange-5,30,40],dtype=np.uint8)
    orange_upper = np.array([H_orange+5,230,240],dtype=np.uint8)

    H_white = 125//2
    white_lower = np.array([H_white-10,10,110],dtype=np.uint8)
    white_upper = np.array([H_white+10,30,200],dtype=np.uint8)

    ROI = np.zeros(frame.shape,np.uint8)
    y,x = frame.shape[:2]
    # pts = np.array([[0,y],[x,y],[5*x//8,int(y*0.6)],[4*x//10,int(y*0.6)]])
    pts = np.array([[0,y-100],[x,y-100],[5*x//8,int(y*0.65)],[4*x//10,int(y*0.65)]])
    ROI = cv2.fillPoly(ROI,[pts],(255,255,255))
    ROI_img = cv2.bitwise_and(frame,ROI)

    HSV_ROI = cv2.cvtColor(ROI_img,cv2.COLOR_BGR2HSV)
    thresh_orange = cv2.inRange(HSV_ROI,orange_lower,orange_upper)
    thresh_white = cv2.inRange(HSV_ROI,white_lower,white_upper)
    roads = cv2.add(thresh_orange,thresh_white)
    roads = cv2.morphologyEx(roads, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

    # adge = cv2.Canny(roads,300,400)
    linesP = cv2.HoughLinesP(roads,1,np.pi/180,threshold=10,minLineLength=5,maxLineGap=30)

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

    w,h = frame.shape[:2]
    # roads = cv2.resize(roads,(h//2,w//2))
    frame = cv2.resize(frame, (h//2,w//2))
    cv2.imshow('frame',frame)
    # cv2.imshow('ROI',roads)
    
    if cv2.waitKey(0) == 27:
        break

cv2.release()
cv2.destroyAllWindows()