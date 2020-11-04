import cv2
import numpy as np
from keras import backend as K
from keras.models import model_from_json
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

print('start testing...')
print()

K.set_image_data_format('channels_first')

# load model from json and the weights
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

# loads an image and then detects the symbols and evaluates the answer
def eval_image(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    # cv2.imshow("start",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    

    if img is not None:
        img = ~img
        ret, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        ret, contours, ret = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contour = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        w=28
        h=28
        data = []
        rects = []
        for c in contour:
            x,y,w,h= cv2.boundingRect(c)
            rect=[x,y,w,h]
            rects.append(rect)            
        bool_rect=[]
        for r in rects:
            l=[]
            for rec in rects:
                flag=0
                if rec!=r:
                    if r[0]<(rec[0]+rec[2]+10) and rec[0]<(r[0]+r[2]+10) and r[1]<(rec[1]+rec[3]+10) and rec[1]<(r[1]+r[3]+10):
                        flag=1
                    l.append(flag)
                if rec==r:
                    l.append(0)
            bool_rect.append(l)
        dump_rect=[]
        for i in range(0,len(contour)):
            for j in range(0,len(contour)):
                if bool_rect[i][j]==1:
                    area1=rects[i][2]*rects[i][3]
                    area2=rects[j][2]*rects[j][3]
                    if(area1==min(area1,area2)):
                        dump_rect.append(rects[i])
        final_rect=[i for i in rects if i not in dump_rect]
        for r in final_rect:
            x=r[0]
            y=r[1]
            w=r[2]
            h=r[3]
            im_crop =thresh[y:y+h+10,x:x+w+10]
            im_resize = cv2.resize(im_crop,(28,28))

            # cv2.imshow("rect",im_resize)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            im_resize=np.reshape(im_resize,(1,28,28))
            data.append(im_resize)

    equation=''
    for i in range(len(data)):
        data[i]=np.array(data[i])
        data[i]=data[i].reshape(1,1,28,28)
        result=loaded_model.predict_classes(data[i])

        if(result[0]==0):
            equation=equation+'0'
        if(result[0]==1):
            equation=equation+'1'
        if(result[0]==2):
            equation=equation+'2'
        if(result[0]==3):
            equation=equation+'3'
        if(result[0]==4):
            equation=equation+'4'
        if(result[0]==5):
            equation=equation+'5'
        if(result[0]==6):
            equation=equation+'6'
        if(result[0]==7):
            equation=equation+'7'
        if(result[0]==8):
            equation=equation+'8'
        if(result[0]==9):
            equation=equation+'9'
        if(result[0]==10):
            equation=equation+'+'
        if(result[0]==11):
            equation=equation+'-'
    print(f'{equation} = {eval(equation)}') 

eval_image('ml\\test_img\\5.png')
eval_image('ml\\test_img\\test1.png')
eval_image('ml\\test_img\\test2.png')

print()
print('testing finished...')