import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread('written.jpg')

b,g,r = cv2.split(img)           # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb


# Denoising
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

b,g,r = cv2.split(dst)           # get b,g,r
rgb_dst = cv2.merge([r,g,b])     # switch it to rgb

plt.subplot(2,2,1), plt.imshow(rgb_img,cmap = 'gray')
plt.title('Original Noisy Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2), plt.imshow(rgb_dst,cmap = 'gray')
plt.title('Noise less image'), plt.xticks([]), plt.yticks([])

im = Image.fromarray(rgb_dst)
im.save("img2.jpg")


#Binarization
img1 = cv2.imread('img2.jpg',0)

th3 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

ret, imgf = cv2.threshold(th3, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
plt.subplot(2,2,3), plt.imshow(img1,cmap = 'gray')
plt.title('Noiseless Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4), plt.imshow(imgf,cmap = 'gray')
plt.title('Otsu thresholding'), plt.xticks([]), plt.yticks([])
plt.show()
