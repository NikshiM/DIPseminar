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
You can add two images with the OpenCV function, cv. add(), or simply by the numpy operation res = img1 + img2.
The function mean calculates the mean value M of array elements, independently for each channel, and return it:" This mean it should return you a scalar for each layer of you image

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

The append() method in python adds a single item to the existing list.
listdir() method in python is used to get the list of all files and directories in the specified directory.


## OUTPUT:-
![output](https://user-images.githubusercontent.com/72375228/104420807-728e7880-55a0-11eb-9dc6-9f459dbad8c4.PNG)



## 4.Write a program to convert color image into gray scale and binary image.
Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.
A binary image is a monochromatic image that consists of pixels that can have one of exactly two colors, usually black and white.


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

## OUTPUT:- 
![i5](https://user-images.githubusercontent.com/72375228/104286388-fbdb7780-54da-11eb-8c15-a085a14732f8.PNG)

## 5.Write a program to convert color image into different color space.

Color spaces are a way to represent the color channels present in the image that gives the image that particular hue
BGR color space: OpenCV’s default color space is RGB. 
HSV color space: It stores color information in a cylindrical representation of RGB color points. It attempts to depict the colors as perceived by the human eye. Hue value varies from 0-179, Saturation value varies from 0-255 and Value value varies from 0-255. 
LAB color space :
L – Represents Lightness.A – Color component ranging from Green to Magenta.B – Color component ranging from Blue to Yellow.
The HSL color space, also called HLS or HSI, stands for:Hue : the color type Ranges from 0 to 360° in most applications 
Saturation : variation of the color depending on the lightness.
Lightness :(also Luminance or Luminosity or Intensity). Ranges from 0 to 100% (from black to white).
YUV:Y refers to the luminance or intensity, and U/V channels represent color information. This works well in many applications because the human visual system perceives intensity information very differently from color information.


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
COLOR_BGR2HSV:Conver the color BGR to HSV.
COLOR_BGR2HLS:Conver the color BGR to HLS.
COLOR_BGR2LAB:Conver the color BGR to LAB
COLOR_BGR2YUV:Conver the color BGR to YUV.


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

## 7.Find the sum of all neighborhood values of the matrix.

Given a  matrix and a set of cell indexes e.g., an array of (i, j) where i indicates row and j column. For every given cell index (i, j), finding sums of all matrix elements except the elements present in i’th row and/or j’th column.

import numpy as np
M = [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]] 
M = np.asarray(M)
N = np.zeros(M.shape)
def sumNeighbors(M,x,y):
    l = []
    for i in range(max(0,x-1),x+2): # max(0,x-1), such that no negative values in range() 
        for j in range(max(0,y-1),y+2):
            try:
                t = M[i][j]
                l.append(t)
            except IndexError: # if entry doesn't exist
                pass
    return sum(l)-M[x][y] # exclude the entry itself
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        N[i][j] = sumNeighbors(M, i, j)
print ("Original matrix:\n", M)
print ("Summed neighbors matrix:\n", N)

The append() method in python adds a single item to the existing list.
The function "shape" returns the shape of an array. The shape is a tuple of integers. These numbers denote the lengths of the corresponding array dimension. 

## OUTPUT:
Original matrix:
 [[1 2 3]
 [4 5 6]
 [7 8 9]]
Summed neighbors matrix:
 [[11. 19. 13.]
 [23. 40. 27.]
 [17. 31. 19.]]
 
 ## 8.Write a c++ program to perform Operator overloading.
 C++ has the ability to provide the operators with a special meaning for a data type, this ability is known as operator overloading.
 Operator overloading is a compile-time polymorphism in which the operator is overloaded to provide the special meaning to the user-defined data type. Operator overloading is used to overload or redefines most of the operators available in C++. It is used to perform the operation on the user-defined data type.
 
 #include <iostream>
using namespace std;
class matrix
{
 int r1, c1, i, j, a1;
 int a[10][10];

public:int get()
 {
  cout << "Enter the row and column size for the  matrix\n";
  cin >> r1;
  cin >> c1;
   cout << "Enter the elements of the matrix\n";
  for (i = 0; i < r1; i++)
  {
   for (j = 0; j < c1; j++)
   {
    cin>>a[i][j];

   }
  }
 
 
 };
 void operator+(matrix a1)
 {
 int c[i][j];
  
   for (i = 0; i < r1; i++)
   {
    for (j = 0; j < c1; j++)
    {
     c[i][j] = a[i][j] + a1.a[i][j];
    }
   
  }
  cout<<"addition is\n";
  for(i=0;i<r1;i++)
  {
   cout<<" ";
   for (j = 0; j < c1; j++)
   {
    cout<<c[i][j]<<"\t";
   }
   cout<<"\n";
  }

 };

  void operator-(matrix a2)
 {
 int c[i][j];
  
   for (i = 0; i < r1; i++)
   {
    for (j = 0; j < c1; j++)
    {
     c[i][j] = a[i][j] - a2.a[i][j];
    }
   
  }
  cout<<"subtraction is\n";
  for(i=0;i<r1;i++)
  {
   cout<<" ";
   for (j = 0; j < c1; j++)
   {
    cout<<c[i][j]<<"\t";
   }
   cout<<"\n";
  }
 };

 void operator*(matrix a3)
 {
  int c[i][j];

  for (i = 0; i < r1; i++)
  {
   for (j = 0; j < c1; j++)
   {
    c[i][j] =0;
    for (int k = 0; k < r1; k++)
    {
     c[i][j] += a[i][k] * (a3.a[k][j]);
    }
  }
  }
  cout << "multiplication is\n";
  for (i = 0; i < r1; i++)
  {
   cout << " ";
   for (j = 0; j < c1; j++)
   {
    cout << c[i][j] << "\t";
   }
   cout << "\n";
  }
 };

};

int main()
{
 matrix p,q;
 p.get();
 q.get();
 p + q;
 p - q;
 p * q;
return 0;
}

## OUTPUT:
Enter the row and column size for the  matrix
2
2
Enter the elements of the matrix
6
7
5
8
Enter the row and column size for the  matrix
2
2
Enter the elements of the matrix
2
3
1
4
addition is
 8      10
 6      12
subtraction is
 4      4
 4      4
multiplication is
 19     46
 18     47

## 9.Develop a program to find the neighbor of each element in the matrix

import numpy as np
ini_array = np.array([[1, 2,5, 3], [4,5, 4, 7], [9, 6, 1,0]])
print("initial_array : ", str(ini_array));
def neighbors(radius, rowNumber, columnNumber):
    return[[ini_array[i][j]if i >= 0 and i < len(ini_array) and j >= 0 and j < len(ini_array[0]) else 0
            for j in range(columnNumber-1-radius, columnNumber+radius)]
           for i in range(rowNumber-1-radius, rowNumber+radius)]
neighbors(1,2,2)


## OUTPUT:
initial_array :  [[1 2 5 3]
                  [4 5 4 7]
                  [9 6 1 0]]
                  
   [[2, 5, 3],
    [5, 4, 7], 
    [6, 1, 0]]
   
















