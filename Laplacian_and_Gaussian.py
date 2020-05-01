
import numpy as np
import math
from cv2 import cv2
def lapl_pyramid(gauss_pyr):
  output = []
  k = len(gauss_pyr)
  for i in range(0,k-1):
    gu = gauss_pyr[i]
    egu = iexpand(gauss_pyr[i+1])
    if egu.shape[0] > gu.shape[0]:
       egu = np.delete(egu,(-1),axis=0)
    if egu.shape[1] > gu.shape[1]:
      egu = np.delete(egu,(-1),axis=1)
    output.append(gu - egu)
  output.append(gauss_pyr.pop())
  return output

  '''create a gaussain pyramid of a given image'''
def gauss_pyramid(image, levels):
  output = []
  output.append(image)
  tmp = image
  for i in range(0,levels):
    tmp = ireduce(tmp)
    output.append(tmp)
  return output

  '''expand image by factor of 2'''
def iexpand(image):
  out = None
  h= np.array([1,4,6,4,1])/16
  filt= (h.T).dot(h)
  outimage = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)
  outimage[::2,::2]=image[:,:]
  out = cv2.filter2D(outimage,cv2.CV_64F,filt)
  return out

  '''reduce image by 1/2'''
def ireduce(image):
  out = None
  h= np.array([1,4,6,4,1])/16
  filt= (h.T).dot(h)
  outimage = cv2.filter2D(image,cv2.CV_64F,filt)
  out = outimage[::2,::2]
  return out

'''Reconstruct the image based on its laplacian pyramid.'''
def collapse(lapl_pyr):
  output = None
  
  output = np.zeros((lapl_pyr[0].shape[0],lapl_pyr[0].shape[1]), dtype=np.float64)
  for i in range(len(lapl_pyr)-1,0,-1):
    lap = iexpand(lapl_pyr[i])
    lapb = lapl_pyr[i-1]
    if lap.shape[0] > lapb.shape[0]:
      lap = np.delete(lap,(-1),axis=0)
    if lap.shape[1] > lapb.shape[1]:
      lap = np.delete(lap,(-1),axis=1)
    tmp = lap + lapb
    
    output = tmp
  return output