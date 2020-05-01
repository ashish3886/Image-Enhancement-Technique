import numpy as np
from cv2 import cv2

def adapt_hist_equilization(img):

#img = cv2.imread("test_001.jpg")
    R, G, B = cv2.split(img)

    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)

    equ = cv2.merge((output1_R, output1_G, output1_B))
    res = np.hstack((img,equ)) #stacking images side-by-side
    return res
    #cv2.imshow('clahe_2.jpg',res)
    #cv2.waitKey(0)
