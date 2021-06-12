import cv2
import matplotlib.pyplot as plt
import numpy

cap = cv2.VideoCapture('roads_line.mp4')

ret,frame = cap.read()

cv2.imshow('image',frame)
cv2.waitKey(0)

cv2.release()
cv2.destroyAllWindows()