#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from mpl_toolkits.axes_grid1 import ImageGrid
import random
import shutil
from numba import cuda 
device = cuda.get_current_device()




# In[2]:


data_path = '../Data/full dataset'
data_path_1000 = '../Data/Dataset 1000/train'
data_path_800 = '../Data/Dataset 800/train'
data_path_200 = '../Data/Dataset 200/train'
val_data_path = '../Data/validation dataset/'
figures_output_path = '../Outputs/figures'
csv_outputs ='../Outputs/csv'
models_output_path = '../Models'
model_checkpoints_path = '../Models/checkpoints'


# In[3]:
device.reset()


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# In[4]:


# Data parameters
batch_size = 4
img_height = 224
img_width = 224
n_classes=5



# Model Parameters
lr_sched_trigger=5

# In[5]:


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_path_200,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  val_data_path,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  os.path.join(data_path,'test'),
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[6]:


def preprocess(image, label):
    normalise = tf.cast(image, tf.float32) / 255
    
    final_image = normalise
    return final_image, label

def xception_preprocess(image, label):
    normalise = tf.cast(image, tf.float32) / 255
    final_image = tf.keras.applications.xception.preprocess_input(normalise)
    final_image = normalise
    return final_image, label

def lr_scheduler(epoch, lr):
  if epoch<lr_sched_trigger:
    return lr
  elif epoch%10 ==0| epoch ==lr_sched_trigger :
    lr= lr * tf.math.exp(-0.1)
    return lr
  else:
    return lr


optimizer1= tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

optimizer2= tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

###########

#ResNet 18


###########



train_ds = train_ds.map(preprocess).prefetch(1)
val_ds = val_ds.map(preprocess).prefetch(1)
test_ds = test_ds.map(preprocess).prefetch(1)


# In[8]:


AUTOTUNE = tf.data.AUTOTUNE



train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)




# Model Callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(models_output_path, 'Resnet18.h5'),
  save_best_only=True) 
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)



base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width,3))
flatten = tf.keras.layers.Flatten()(base_model.output)
fc1 = tf.keras.layers.Dense(1024, 'relu')(flatten)
do1 = tf.keras.layers.Dropout(.2)(fc1)
#pool = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation='softmax')(do1)
model=tf.keras.Model(inputs=base_model.input, outputs=output)


# In[10]:


for layer in base_model.layers:
    layer.trainable=False


# In[11]:




model.compile(
  optimizer=optimizer1,
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy' ])


# In[ ]:


history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=10,
  callbacks=[checkpoint_cb, early_stopping_cb, lr_cb]
)

full_training_hist = pd.DataFrame(history.history)
## Allow fine tuning
for layer in base_model.layers[-6:]:
    layer.trainable=True




model.compile(
  optimizer=optimizer2,
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


# In[ ]:


history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5,
  callbacks=[checkpoint_cb, early_stopping_cb, lr_cb]
)

full_training_hist.append(history.history)
# In[ ]:


# summarize history for accuracy
plt.subplot(1,2,1)
plt.plot(full_training_hist['accuracy'])
plt.plot(full_training_hist['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.subplot(1,2,2)
# summarize history for loss
plt.plot(full_training_hist['loss'])
plt.plot(full_training_hist['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.savefig('Xception 200 images per class.jpg')



device.reset()


"""

###########
"""
#Xception

"""
##########

train_ds = train_ds.map(xception_preprocess).prefetch(1)
val_ds = val_ds.map(xception_preprocess).prefetch(1)
test_ds = test_ds.map(xception_preprocess).prefetch(1)


# In[8]:


AUTOTUNE = tf.data.AUTOTUNE

#train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[9]:


base_model = tf.keras.applications.xception.Xception(weights='imagenet', include_top=False)
pool = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation='softmax')(pool)
model=tf.keras.Model(inputs=base_model.input, outputs=output)


# In[10]:


for layer in base_model.layers:
    layer.trainable=False


# In[11]:


optimizer= tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(
  optimizer=optimizer,
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy' ])


# In[ ]:


history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=10
)


## Allow fine tuning
for layer in base_model.layers:
    layer.trainable=True


optimizer= tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(
  optimizer=optimizer,
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


# In[ ]:


history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5
)


# In[ ]:


# summarize history for accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.subplot(1,2,2)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.savefig('Xception 200 images per class.jpg')


# In[ ]:


"""

