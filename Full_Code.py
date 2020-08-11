# Used Udacity Simulator to collect data, while collecting I made sure to get equal amount of data for the right turn and left turn movement of the steering
# I believe 3 laps for right and left individually would be enough data 
# After that I have saved the data in github, to use the link for my training
# For using in the code, write: '! git clone (URL of the file)'

import os
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
import cv2
import ntpath #To vary the path
import random
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.image as mpimg

# 1. INITIALISATION

# In this repository I have just used the following columns of the data: 
# 1)center 2)left 3)right 4)steering 5)throttle 6)reverse 7)speed
datadir = 'Track'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
pd.set_option('display.max_colwidth', -1) #for printing whole data
data.head() 

def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail

data['center'] = data['center'].apply(path_leaf) #As data is a DataFrame, we can apply on all the column values 
data['left'] = data['left'].apply(path_leaf) 
data['right'] = data['right'].apply(path_leaf)
data.head()

# 2. VISUALISATION AND PREPROCESSING OF DATA

num_of_bins = 25
samples_per_bins = 350 #The frquency at 0 is veryyy high, this will make our model biased, to avoid that, we will initialise a threshold
hist, bins = np.histogram(data['steering'], num_of_bins) 
center = (bins[:-1] + bins[1:]) * 0.5   #for making it centered to zero, 0.5 is becoz the operation will double the values
print(center) #we get both -ve and +ve because of the left and right turns
print(plt.bar(center, hist, width = 0.05))
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bins, samples_per_bins))

print(len(data))
remove_list = []
for j in range(num_of_bins):
  list_ =[]
  for i in range(len(data['steering'])):
    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
      list_.append(i)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bins:]
  remove_list.extend(list_)
print(len(remove_list))
data.drop(data.index[remove_list], inplace = True)
print(data)
hist, _ = np.histogram(data['steering'], num_of_bins) 
print(plt.bar(center, hist, width = 0.05))
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bins, samples_per_bins))

print(data.iloc[0])
def load_img_steering(datadir, df):
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i] #specific row of data
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    image_path.append(os.path.join(datadir, center.strip())) #We are caring only about center images for now and strip is used to remove spaces between the path
    steering.append(float(indexed_data[3]))
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

image_paths, steerings = load_img_steering(datadir + '/IMG', data)

X_train, X_val, y_train, y_val = train_test_split(image_paths, steerings, test_size = 0.2, random_state = 6)
print("Training Samples: {}\n Valid Samples: {}".format(len(X_train), len(X_val)))

fig, axes = plt.subplots(1, 2, figsize = (12,4))
axes[0].hist(y_train, bins = num_of_bins, width = 0.05, color = 'blue')
axes[0].set_title('Training Set')
axes[1].hist(y_val, bins = num_of_bins, width = 0.05, color = 'red')
axes[1].set_title('Validation Set')

def img_preprocess(img):
  img = mpimg.imread(img)
  img = img[60:140,:,:] #To remove the unneccessary part
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) # Y represents Luminosity(Brightness), UV chromiens ..it adds color to img
  # fig, axes = plt.subplots(1, 1, figsize = (15,10))
  # axes.imshow(img)  
  img = cv2.GaussianBlur(img, (3,3), 0)
  img = cv2.resize(img, (200,66)) #Not necessary it will just keep it consistent
  img = img/255
  return img
  
image = image_paths[100]
original_image = mpimg.imread(image)
preprocessed_image = img_preprocess(image)
fig, axes = plt.subplots(1, 2, figsize = (15,10))
fig.tight_layout() #So Axes won't overlap
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[1].imshow(preprocessed_image)
axes[1].set_title('Preprocessed Image')

X_train = np.array(list(map(img_preprocess, X_train)))
X_val = np.array(list(map(img_preprocess, X_val)))

plt.imshow(X_train[random.randint(0, len(X_val) - 1)])
plt.axis("off")
print(X_val.shape)

# 3. DEFINING THE MODEL

def nvidia_model():
  model = Sequential()
  model.add(Convolution2D(24, (5, 5), strides = (2, 2), input_shape = (66, 200, 3), activation= 'elu'))
  model.add(Convolution2D(36, (5, 5), strides = (2, 2), activation = 'elu'))
  model.add(Convolution2D(48, (5, 5), strides = (2, 2), input_shape = (66, 200, 3), activation= 'elu'))
  model.add(Convolution2D(64, (3, 3), activation = 'elu'))
  model.add(Convolution2D(64, (3, 3), activation = 'elu'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense( 100, activation = 'elu'))
  model.add(Dropout(0.5))
  model.add(Dense( 50, activation = 'elu'))
  model.add(Dropout(0.5))
  model.add(Dense( 10, activation = 'elu'))
  model.add(Dropout(0.5))
  model.add(Dense( 1))
  optimizer = Adam(lr = 1e-3)
  model.compile(loss = 'mse', optimizer = optimizer)
  return model
  
model = nvidia_model()
print(model.summary())

history = model.fit(X_train, y_train, epochs = 30, validation_data = (X_val, y_val), batch_size = 100, verbose = 1, shuffle = 1)

# 4. EVALUATING THE MODEL

#Loss Graph vs Epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epochs')

#To test it using Udacity Simulator
model.save('model.h5')

from google.colab import files
files.download('model.h5')
