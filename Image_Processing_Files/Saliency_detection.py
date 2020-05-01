import cv2
import numpy as np
from skimage import color
from scipy.ndimage import gaussian_filter
def saliencyDetection(img):
    gfrgb = gaussian_filter(img, order = 0 ,sigma=1)
    lab= color.rgb2lab(gfrgb)
    #Compute Lab average values
    l = np.double(lab[:,:,0])
    a = np.double(lab[:,:,1])
    b = np.double(lab[:,:,2])
    lm = np.mean(np.mean(l))
    am = np.mean(np.mean(a))
    bm = np.mean(np.mean(b))
    #Finally compute the saliency map and return.
    sm = np.square(l-lm)+ np.square(a-am) + np.square((b-bm))
    return sm
def main():
   # im=cv2.imread("input_1.jpg")
   # sm=saliencyDetection(im)
   # print ("nm",sm)
   # cv2.imshow("after", sm)
   # cv2.waitKey(0)
if __name__ == '__main__':
   # main()