# DIPseminar
## 1.Develop a program to display grayscale image using read and write operation.
Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.
Importance of grayscaling :-
Dimension reduction: For e.g. In RGB images there are three color channels and has three dimensions while grayscaled images are single dimensional.
Reduces model complexity: Consider training neural article on RGB images of 10x10x3 pixel.The input layer will have 300 input nodes. On the other hand, the same neural network will need only 100 input node for grayscaled images.
For other algorithms to work: There are many algorithms that are customized to work only on grayscaled images e.g. Canny edge detection function pre-implemented in OpenCV library works on Grayscaled images only.
## PROGRAM1:

import cv2
import numpy as np
image = cv2.imread('img20.jpg')
image = cv2.resize(image, (0, 0), None, .25, .25)
grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
numpy_horizontal = np.hstack((image, grey_3_channel))
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
cv2.imshow('flower', numpy_horizontal_concat)
cv2.imwrite('nik.jpg',grey)
cv2.waitKey()

cv2.resize() method refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications. It helps in reducing the number of pixels from an image 
cv2.cvtColor() method is used to convert an image from one color space to another. 
np.hstack() function is used to stack the sequence of input arrays horizontally (i.e. column wise) to make a single array.
np.concatenate: Concatenation refers to joining. This function is used to join two or more arrays of the same shape along a specified axis.
cv2.imwrite() method is used to save an image to any storage device. This will save the image according to the specified format in current working directory.


## OUTPUT:
![i1](https://user-images.githubusercontent.com/72375228/104284807-a7cf9380-54d8-11eb-8bc4-7e0961e2aaf3.PNG)

## 2. Develop a program to perform linear transformation on image. (Scaling and rotation) 
## //Scaling
Image resizing refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications. It helps in reducing the number of pixels from an image 

import cv2 as c
img=c.imread("img3.jpg")
c.imshow('image',img)
nimg=c.resize(img,(0,0),fx=0.50,fy=0.50)
c.imshow("Result",nimg)
c.waitKey(0)

cv2.resize() method refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications. It helps in reducing the number of pixels from an image 
 imshow() function in pyplot module of matplotlib library is used to display data as an image

OUTPUT:-
![i1](https://user-images.githubusercontent.com/72375228/104285244-52e04d00-54d9-11eb-9c41-cfba842ea83a.PNG)

## //Rotation
Image rotation is a common image processing routine used to rotate images at any desired angle. This helps in image reversal, flipping, and obtaining an intended view of the image. Image rotation has applications in matching, alignment, and other image-based algorithms. OpenCV is a well-known library used for image processing.

import cv2 
import numpy as np 
img = cv2.imread('img22.jfif') 
(rows, cols) = img.shape[:2] 
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 120, 1) 
res = cv2.warpAffine(img, M, (cols, rows)) 
cv2.imshow('image', img)
cv2.waitKey(0) 
cv2.imshow('result',res) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

cv2.getRotationMatrix2D Perform the counter clockwise rotation
warpAffine() function is the size of the output image, which should be in the form of (width, height). Remember width = number of columns, and height = number of rows.

## OUTPUT:-
![i3](https://user-images.githubusercontent.com/72375228/104285443-a357aa80-54d9-11eb-99fc-76fc71446222.PNG)

## 3. Develop a program to find sum and mean of a set of images.Create n number of images and read the directory and perform operation.

import cv2
import os
path = 'C:\images'
imgs = []

files = os.listdir(path)
for file in files:
    filepath=path+"\\"+file
    imgs.append(cv2.imread(filepath))
i=0
im = []
for im in imgs:
    #cv2.imshow(files[i],imgs[i])
    im+=imgs[i]
    i=i+1
cv2.imshow("sum of five pictures",im)
meanImg = im/len(files)
cv2.imshow("mean of five pictures",meanImg)
cv2.waitKey(0)


## OUTPUT:-
![output](https://user-images.githubusercontent.com/72375228/104420807-728e7880-55a0-11eb-9dc6-9f459dbad8c4.PNG)



## 4.Write a program to convert color image into gray scale and binary image.
Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.
A binary image is a monochromatic image that consists of pixels that can have one of exactly two colors, usually black and white

import cv2
image=cv2.imread("img19.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(tresh,blackAndWhiteImage)=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow("gray",gray)
cv2.imshow("BINARY",blackAndWhiteImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.threshold works as, if pixel value is greater than a threshold value, it is assigned one value (may be white), else it is assigned another value (may be black). 
destroyAllWindows() simply destroys all the windows we created. To destroy any specific window, use the function cv2. destroyWindow() where you pass the exact window name.

OUTPUT:- 
![i5](https://user-images.githubusercontent.com/72375228/104286388-fbdb7780-54da-11eb-8c15-a085a14732f8.PNG)

## 5.Write a program to convert color image into different color space.

Color spaces are a way to represent the color channels present in the image that gives the image that particular hue
BGR color space: OpenCV’s default color space is RGB. 
HSV color space: It stores color information in a cylindrical representation of RGB color points. It attempts to depict the colors as perceived by the human eye. Hue value varies from 0-179, Saturation value varies from 0-255 and Value value varies from 0-255. 
LAB color space :
L – Represents Lightness.
A – Color component ranging from Green to Magenta.
B – Color component ranging from Blue to Yellow.

import cv2
image=cv2.imread("img20.jpg")
cv2.imshow("old",image)
cv2.waitKey()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV",hsv)
cv2.waitKey(0)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("LAB",lab)
cv2.waitKey(0)
hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
cv2.imshow("HLS",hls)
cv2.waitKey(0)
yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
cv2.imshow("YUV",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.cvtColor() method is used to convert an image from one color space to another. 

## OUTPUT:-
![ii](https://user-images.githubusercontent.com/72375228/104288153-73120b00-54dd-11eb-9634-4c29a761b044.PNG)

## 6.Develop a program to create an image from 2D array.
2D array can be defined as an array of arrays. The 2D array is organized as matrices which can be represented as the collection of rows and columns. However, 2D arrays are created to implement a relational database look alike data structure.

import numpy as np
from PIL import Image
import cv2 as c 
array = np.zeros([100, 200, 3], dtype=np.uint8)
array[:,:100] = [150, 128, 0] #Orange left side
array[:,100:] = [0, 0, 255]   #Blue right side
img = Image.fromarray(array)
img.save('flower.jpg')
img.show()

numpy.zeros() function returns a new array of given shape and type, with zeros.
Image.fromarray(array) is creating image object of above array

## OUTPUT:-
![i6](https://user-images.githubusercontent.com/72375228/104289209-e0726b80-54de-11eb-95ef-ea42ae76b383.PNG)









