import os
import cv2
import numpy as np
import pandas as pd

print('preprocessing data...')

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
data = load('img\\0')
for i in range(0,len(data)):
    data[i]=np.append(data[i],['0'])
print(len(data))


df = pd.DataFrame(data, index=None)
df.to_csv('training_data.csv', index=False)

print('preprocessing finished...')
