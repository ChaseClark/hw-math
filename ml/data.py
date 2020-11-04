import os
import cv2
import numpy as np
import pandas as pd

print('preprocessing data...')
print()

# function to load the test images from img folder
def load(path):
    img_data = []
    # loop through every img in folder
    for file in os.listdir(path):
        # store the image as grayscale
        img = cv2.imread(os.path.join(path,file),cv2.IMREAD_GRAYSCALE)
        # perform a bitwise inversion of black/white
        # this makes the background blank and the symbol white, which helps opencv
        img = ~img
        # cv2.imshow(file,img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if img is not None:
            # thresholds https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
            ret, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            # contours https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html
            ret,contours,ret=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            contour = sorted(contours, key=lambda ctr:cv2.boundingRect(ctr)[0])
            w=int(28)
            h=int(28)
            maxi=0
            # bounding rect https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html 
            for c in contour:
                x,y,w,h=cv2.boundingRect(c)
                maxi=max(w*h,maxi)
                if maxi==w*h:
                    x_max=x
                    y_max=y
                    w_max=w
                    h_max=h
            im_crop= thresh[y_max:y_max+h_max+10, x_max:x_max+w_max+10]
            # make all images the same size 28 by 28
            im_resize = cv2.resize(im_crop,(28,28))
            # flatten the image so we can store it in a csv file
            im_resize=np.reshape(im_resize,(784,1))
            img_data.append(im_resize)
    return img_data            

# for each symbol that we want to process, load all images of that symbol and append the flattened image to our np array

data = []
# 0
data = load('ml\img\\0')
for i in range(0,len(data)):
    data[i]=np.append(data[i],['0'])
print(len(data))

# 1
data1 = load('ml\img\\1')
for i in range(0,len(data1)):
    data1[i]=np.append(data1[i],['1'])
data = np.concatenate((data,data1))
print(len(data))

# 2
data2 = load('ml\img\\2')
for i in range(0,len(data2)):
    data2[i]=np.append(data2[i],['2'])
data = np.concatenate((data,data2))
print(len(data))

# 3
data3 = load('ml\img\\3')
for i in range(0,len(data3)):
    data3[i]=np.append(data3[i],['3'])
data = np.concatenate((data,data3))
print(len(data))

# 4
data4 = load('ml\img\\4')
for i in range(0,len(data4)):
    data4[i]=np.append(data4[i],['4'])
data = np.concatenate((data,data4))
print(len(data))

# 5
data5 = load('ml\img\\5')
for i in range(0,len(data5)):
    data5[i]=np.append(data5[i],['5'])
data = np.concatenate((data,data5))
print(len(data))

# 6
data6 = load('ml\img\\6')
for i in range(0,len(data6)):
    data6[i]=np.append(data6[i],['6'])
data = np.concatenate((data,data6))
print(len(data))

# 7
data7 = load('ml\img\\7')
for i in range(0,len(data7)):
    data7[i]=np.append(data7[i],['7'])
data = np.concatenate((data,data7))
print(len(data))

# 8
data8 = load('ml\img\\8')
for i in range(0,len(data8)):
    data8[i]=np.append(data8[i],['8'])
data = np.concatenate((data,data8))
print(len(data))

# 9
data9 = load('ml\img\\9')
for i in range(0,len(data9)):
    data9[i]=np.append(data9[i],['9'])
data = np.concatenate((data,data9))
print(len(data))

# assign + to 10
data10 = load('ml\img\\+')
for i in range(0,len(data10)):
    data10[i]=np.append(data10[i],['10'])
data = np.concatenate((data,data10))
print(len(data))

# -
data11 = load('ml\img\\-')
for i in range(0,len(data11)):
    data11[i]=np.append(data11[i],['11'])
data = np.concatenate((data,data11))
print(len(data))



df = pd.DataFrame(data, index=None)
df.to_csv('ml\\training_data.csv', index=False)
print()
print('preprocessing finished...')
