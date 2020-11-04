import pandas as pd
import numpy as np
from keras import backend as K
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

# injection method to fix issue with mismatching tf and keras versions 
# https://github.com/keras-team/keras/issues/13684
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

K.set_image_data_format('channels_first')

print('training starting...')
print()

df = pd.read_csv('ml\\training_data.csv', index_col=False)
# last col is labels
# todo do not hardcode this value
col_len = len(df.columns)-1
# print(col_len)
labels = df[[str(col_len)]]
labels = np.array(labels)
unique = np.unique(labels)
# get how many unique symbols we are processing
num_classes = len(unique)
# remove label col
df.drop(df.columns[[col_len]],axis=1,inplace=True)
# number of symbols we are processing
cat=to_categorical(labels,num_classes=num_classes)

df_len = len(df)

arr = []
for i in range(df_len):
    arr.append(np.array(df[i:i+1]).reshape(1,28,28))

# create model
model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape=(1 , 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit model
model.fit(np.array(arr), cat, epochs=10, batch_size=200, shuffle=True, verbose=1)


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")

print()
print('training finished...')