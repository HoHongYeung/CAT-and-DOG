# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 23:20:44 2022

@author: user
"""

# Main libraries
import pandas as pd
from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Network
from tensorflow.keras import models
from tensorflow.keras import layers

# For data directories
def extract_files(dir):
    '''
    This function for extracting the files.
    INPUT:
    dir: str. The main file direction.
    OUTPUT:
    Extracted files.
    '''
    with ZipFile(dir,'r') as zip:
        zip.extractall()

def copy_files(rng,path1,name,path2):
    '''
    This function for copying the files from the source to destination.
    INPUT:
    rng: list . A list that contains the ranges.
    path1: str. The data directory.
    name: str. The animal name.
    path2: str. The destination directory
    '''
    fnames = [f'{name}.{idx}.jpg' for idx in range(rng[0],rng[1])]
    for fname in fnames:
        src = os.path.join(path1,fname)
        dst = os.path.join(path2,fname)
        shutil.copyfile(src,dst)
        
# Extracting the data
train_dir = 'd:/Users/user/Desktop/cat and dog/train1.zip'
test_dir = 'd:/Users/user/Desktop/cat and dog/test11.zip'
extract_files(train_dir)
extract_files(test_dir)
print('Extraction process is done')


# The main path
path = '/Users/user/Desktop/cat and dog'

# Training directory
train_dir = os.path.join(path,'Train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)


# Validation directory
val_dir = os.path.join(path,'Val')
if not os.path.exists(val_dir):
    os.mkdir(val_dir)


# Cats training directory
train_cats_dir = os.path.join(train_dir,'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)


# Cats validation directory
val_cats_dir = os.path.join(val_dir,'cats')
if not os.path.exists(val_cats_dir):
    os.mkdir(val_cats_dir)


# Dogs training directory
train_dogs_dir = os.path.join(train_dir,'dogs')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)


# Dogs validation directory
val_dogs_dir = os.path.join(val_dir,'dogs')
if not os.path.exists(val_dogs_dir):
    os.mkdir(val_dogs_dir)


# Copying training cats and dogs files
copy_files([0,10000],os.path.join(path,'train'),'cat',train_cats_dir)
copy_files([0,10000],os.path.join(path,'train'),'dog',train_dogs_dir)

# Copying validation cats and dogs files
copy_files([10000,12500],os.path.join(path,'train'),'cat',val_cats_dir)
copy_files([10000,12500],os.path.join(path,'train'),'dog',val_dogs_dir)

# Display message
print('Preparing the data directories is done')


# Creating image generator steps
train_data = ImageDataGenerator(rescale=1/255)
validate_data = ImageDataGenerator(rescale=1/255)
train_generator = train_data.flow_from_directory(directory=train_dir,target_size=(128,128),batch_size=20,class_mode='binary')
validate_generator = validate_data.flow_from_directory(directory=val_dir,target_size=(128,128),batch_size=20,class_mode='binary')


# The CNN architecture
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape = (128,128,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

# The model summary
model.summary()

# Fitting the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_generator,steps_per_epoch=1000,epochs=20,validation_data=validate_generator,validation_steps=250,verbose=1)

# Get the results
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(train_acc))

# Plotting the results
fig,axs = plt.subplots(1,2,figsize = (12,7))
axs[0].plot(epochs,train_acc,'bo',label = 'Training Accuracy')
axs[0].plot(epochs,val_acc,'go--',label = 'Validation Accuracy')
axs[0].set_title('Training Acc VS Validation Acc',color='red')
axs[0].set_xlabel('Epochs',color = 'red')
axs[0].set_ylabel('Accuracy',color='red')
axs[0].legend()

axs[1].plot(epochs,train_loss,'bo',label = 'Training Loss')
axs[1].plot(epochs,val_loss,'go--',label = 'Validation Loss')
axs[1].set_title('Training Loss VS Validation Loss',color='red')
axs[1].set_xlabel('Epochs',color = 'red')
axs[1].set_ylabel('Loss',color='red')
axs[1].legend()
plt.show()

# The path
test_dir = 'd:/Users/user/Desktop/cat and dog/test11.zip'
# Extracting data
extract_files(test_dir)

# Create the image generator
test_data_dir = 'd:/Users/user/Desktop/cat and dog/test1'
test_df = pd.DataFrame({'Filename':os.listdir(test_data_dir)})
test_datagen = ImageDataGenerator(rescale=1/255)
test_gen = test_datagen.flow_from_dataframe(test_df,test_data_dir,target_size=(130,130),batch_size=50,x_col='Filename',y_col=None,class_mode=None)



# The first five rows
test_df.head()



# Return the classes
dict((k,v) for k,v in train_gen.class_indices.items())


# Get the prediction
pred = model.predict(test_gen,steps=250)

# Save results
test_df['Pred'] = np.where(pred > 0.5,1,0)


test_df.head()