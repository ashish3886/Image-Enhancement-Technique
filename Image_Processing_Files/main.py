
from simple_color_balance import simplest_cb
from Saliency_detection import saliencyDetection
from Laplacian_and_Gaussian import lapl_pyramid
from Laplacian_and_Gaussian import gauss_pyramid
import math
import numpy as np
from adaptive_histo_equilization import adapt_hist_equilization
from cv2 import cv2

img = cv2.imread("test_001.jpg")
#To get the input 1 from Simple color balance 
image_1 = simplest_cb(img,50) 
#CLAHE 
lab1= cv2.cvtColor(image_1, cv2.COLOR_BGR2LAB)
lab_temp= lab1.copy()
lab2 = adapt_hist_equilization(lab_temp)
image_2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
#For input 1
R1 = np.double(lab1[:,:,0])/255
# calculate laplacian contrast weight
WL1= cv2.Laplacian(R1,cv2.CV_64F)
h= np.array([1,4,6,4,1])/16
filt= (h.T).dot(h)
WC1= cv2.filter2D(R1,cv2.CV_64F,filt)
for i in np.where(WC1>(math.pi/2.75)):
# Formula to calculate laplacian contrast weight
    WC1[i]= math.pi/2.75
WC1= (R1-WC1)*(R1-WC1) 
# calculate the saliency weight for input 1
WS1= saliencyDetection(image_1)
sigma= 0.25
aver= 0.5
# calculate the exposedness weight for input 1
##Formula for the exposdness weight from book 'Progress in Patter recognition,
#Image Analysis,Computer Vision and Application'
WE1= np.exp(-(R1-aver)**2/(2*np.square(sigma)))
# For input2
R2 = np.double(lab2[:,:,0])/255
WL2= cv2.Laplacian(R2,cv2.CV_64F)
h= np.array([1,4,6,4,1])/16
filt= (h.T).dot(h)
WC2= cv2.filter2D(R1,cv2.CV_64F,filt)
for i in np.where(WC2>(math.pi/2.75)):
    WC2[i]= math.pi/2.75
WC2= (R2-WC2)*(R2-WC2)
# calculate the saliency weight for input 2
WS2= saliencyDetection(image_1)
sigma= 0.25
aver= 0.5
# calculate the exposedness weight for input 2
WE2= np.exp(-(R2-aver)**2/(2*np.square(sigma)))
# calculate the normalized weight for both the inputs
W1 = (WL1 + WC1 + WS1 + WE1)/(WL1 + WC1 + WS1 + WE1 + WL2 + WC2 + WS2 + WE2)
W2 = (WL2 + WC2 + WS2 + WE2)/(WL1 + WC1 + WS1 + WE1 + WL2 + WC2 + WS2 + WE2)

#cv2.imshow("weight",np.hstack([W1,W2]))
#Calculate the Gaussian Pyramid for both the inputs
level=5
Weight1= gauss_pyramid(W1,level)
Weight2= gauss_pyramid(W2,level)

R1= None
G1= None
B1= None
R2= None
G2= None
B2= None
(R1,G1,B1)= split_rgb(image_1)
(R2,G2,B2)= split_rgb(image_2)

depth=5 
gauss_pyr_image1r = gauss_pyramid(R1, depth)
gauss_pyr_image1g = gauss_pyramid(G1, depth)
gauss_pyr_image1b = gauss_pyramid(B1, depth)
 
gauss_pyr_image2r = gauss_pyramid(R2, depth)
gauss_pyr_image2g = gauss_pyramid(G2, depth)
gauss_pyr_image2b = gauss_pyramid(B2, depth)
# calculate the laplacian pyramid for input 1

r1  = lapl_pyramid(gauss_pyr_image1r)
g1  = lapl_pyramid(gauss_pyr_image1g)
b1  = lapl_pyramid(gauss_pyr_image1b)
# calculate the laplacian pyramid for input 1
r2 = lapl_pyramid(gauss_pyr_image2r)
g2 = lapl_pyramid(gauss_pyr_image2g)
b2 = lapl_pyramid(gauss_pyr_image2b)
# fusion
R_r = np.array(Weight1)* r1 + np.array(Weight2) * r2
R_g = np.array(Weight1)* g1 + np.array(Weight2) * g2
R_b = np.array(Weight1)* b1 + np.array(Weight2) * b2
R= collapse(R_r)
G= collapse(R_g)
B= collapse(R_b)
# Reconstruction of image to get the final output.
R[R < 0] = 0
R[R > 255] = 255
R = R.astype(np.uint8)
 
G[G < 0] = 0
G[G > 255] = 255
G = G.astype(np.uint8)
 
B[B < 0] = 0
B[B > 255] = 255
B = B.astype(np.uint8)
result = np.zeros(img.shape,dtype=img.dtype)
tmp = []
tmp.append(R)
tmp.append(G)
tmp.append(B)
result = cv2.merge(tmp,result)

cv2.imshow("",np.hstack([img,result]))
cv2.waitKey(0)
cv2.destroyAllWindows()


#split rgb image to its channels'''
def split_rgb(image):
  red = None
  green = None
  blue = None
  (blue, green, red) = cv2.split(image)
  return red, green, blue
