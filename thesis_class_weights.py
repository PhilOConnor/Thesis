#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.axes_grid1 import ImageGrid
import random
import shutil
from numba import cuda 
from sklearn.metrics import confusion_matrix, f1_score
import itertools

#device = cuda.get_current_device()




# In[2]:


data_path = '../Data/full dataset'
data_path_1000 = '../Data/Dataset 1000/train'
data_path_800 = '../Data/Dataset 800/train'
data_path_200 = '../Data/Dataset 200/train'
val_data_path = '../Data/validation dataset/'
figures_output_path = '../Outputs/figures'
csv_output_path ='../Outputs/csv'
models_output_path = '../Models'
model_checkpoints_path = '../Models/checkpoints'


# In[3]:
#device.reset()


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
load_batch_size = 8
batch_size = 8
img_height = 224
img_width = 224
n_classes=5

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_data_path,
    shuffle=True,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=load_batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  os.path.join(data_path,'test'),
  shuffle=True,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=load_batch_size)





AUTOTUNE = tf.data.AUTOTUNE
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

c0 = len(os.listdir(os.path.join(data_path, 'train','0')))
c1 = len(os.listdir(os.path.join(data_path, 'train','1')))
c2 = len(os.listdir(os.path.join(data_path, 'train','2')))
c3 = len(os.listdir(os.path.join(data_path, 'train','3')))
c4 = len(os.listdir(os.path.join(data_path, 'train','4')))

total = c0+c1+c2+c3+c4

weight_for_0 = (1 / c0) * (total / 2.0)
weight_for_1 = (1 / c1) * (total / 2.0)
weight_for_2 = (1 / c2) * (total / 2.0)
weight_for_3 = (1 / c3) * (total / 2.0)
weight_for_4 = (1 / c4) * (total / 2.0)


class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2, 3: weight_for_3, 4: weight_for_4}



for iter_lr, cw in itertools.product([0.001, 0.0001],['cw_fifth', 'cw_half']):

  # Model Parameters
  #learning_rate1 = 0.01
  learning_rate1 = iter_lr
  learning_rate2 = 0.0001

  lr_sched_trigger=10
  epoch1 = 100
  epoch2 = 100

  # In[5]:

  if cw == 'cw_fifth':

    



    train_ds = tf.keras.utils.image_dataset_from_directory(
      os.path.join(data_path,'train'),
      validation_split=0.8,
      subset='training',
      shuffle=True,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=load_batch_size)
  elif cw == 'cw_half':
    train_ds = tf.keras.utils.image_dataset_from_directory(
      os.path.join(data_path,'train'),
      validation_split=0.5,
      subset='training',
      shuffle=True,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=load_batch_size)
  elif cw == 'cw_full':
    train_ds = tf.keras.utils.image_dataset_from_directory(
      os.path.join(data_path,'train'),
      shuffle=True,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=load_batch_size)#
  else:
    break




  


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
    elif epoch%5 ==0 :
      lr= lr * tf.math.exp(-0.1)
      return lr
    else:
      return lr


  optimizer1= tf.keras.optimizers.Adam(
      learning_rate=learning_rate1,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07
  )

  optimizer2= tf.keras.optimizers.Adam(
      learning_rate=learning_rate2,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07
  )



  early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,min_delta=0.01, start_from_epoch=10)
  lr_cb = tf.keras.callbacks.LearningRateScheduler(
      lr_scheduler, verbose=0
  )


  data_augmentations = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode='horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomContrast((0.01, 0.1))
    ])


  ###########

  # VGG16


  ###########


  if iter_lr==0.01 and cw =='cw_fifth':
    pass
  else:
    #train_ds = train_ds.map(preprocess).cache()
    #val_ds = val_ds.map(preprocess).cache()
    #test_ds = test_ds.map(preprocess)

    base_model = tf.keras.applications.VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False, pooling=False)
    base_model.trainable = False


    

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    

    #train_ds = train_ds.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)
    #val_ds = val_ds.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)

    #train_ds = train_ds.map(preprocess).prefetch(buffer_size=AUTOTUNE)
    #val_ds = val_ds.map(preprocess).prefetch(buffer_size=AUTOTUNE)




    # Model Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(models_output_path,'class weights', f'VGG16_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.h5'),
      save_best_only=True) 


    


    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentations(inputs)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x = base_model(x)
    #flatten = tf.k.outputeras.layers.Flatten()(base_model.output)
    pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    fc1 = tf.keras.layers.Dense(4096, 'relu')(pool)
    fc2 = tf.keras.layers.Dense(4096, 'relu')(fc1)
    output = tf.keras.layers.Dense(n_classes, 'softmax')(fc2)
    model=tf.keras.Model(inputs=inputs, outputs=output)


    # In[10]:


    #for layer in base_model.layers:
        #layer.trainable=False


    # In[11]:




    model.compile(
      optimizer=optimizer1,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy' ])


    # In[ ]:


    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epoch1,
      callbacks=[checkpoint_cb, lr_cb],
      class_weight=class_weight
    )

    hist1 = pd.DataFrame(history.history)
    ## Allow fine tuning
    for layer in base_model.layers[-3:]:
        layer.trainable=True




    model.compile(
      optimizer=optimizer2,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])


    # In[ ]:


    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epoch2,
      callbacks=[checkpoint_cb, early_stopping_cb, lr_cb],
      class_weight=class_weight
    )
    hist2 =  pd.DataFrame(history.history)
    hist2.index = hist2.index+epoch1
    full_training_hist = pd.concat([hist1,hist2])
    # In[ ]:


    # summarize history for accuracy
    plt.subplot(1,2,1)
    plt.plot(full_training_hist['accuracy'])
    plt.plot(full_training_hist['val_accuracy'])
    plt.axvline(x = epoch1, color = 'b',linestyle='dashed', label = 'fine-tuning')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val','fine-tuning'], loc='lower right')
    plt.subplot(1,2,2)
    # summarize history for loss
    plt.plot(full_training_hist['loss'])
    plt.plot(full_training_hist['val_loss'])
    plt.axvline(x = epoch1, color = 'b', linestyle='dashed', label = 'fine-tuning')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val','fine-tuning'], loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_output_path, 'graphs','class weights',f'VGG16_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.jpg'))


    preds = model.predict(test_ds, batch_size=batch_size, verbose='auto')
    y = y = np.concatenate([y for x, y in test_ds], axis=0)
    y_hat = preds.argmax(axis=1)
    print(f"Sample output: {list(zip(y[:10], y_hat[:10]))}")

    f1 = f1_score(y, y_hat, average='weighted')

    cm = confusion_matrix(y, y_hat, labels=[0,1,2,3,4])

    print(f"F1 score: {f1}")
    print(f"Confusion Matrix: {cm}")
    f1_df = pd.DataFrame(data={"F1 score":f1}, index=[0])
    cm_df = pd.DataFrame(cm,columns = ['y_hat 0', 'y_hat 1','y_hat 2','y_hat 3','y_hat 4'])






    with pd.ExcelWriter(os.path.join(csv_output_path,'class weights',f'VGG16_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.xlsx'),
                        mode='w') as writer:  

        f1_df.to_excel(writer, startrow=0)
        cm_df.to_excel(writer, startrow=2)
        full_training_hist.to_excel(writer, sheet_name='model history')

  #device.reset()

  
  ###########

  # ResNet50


  ###########
  optimizer1= tf.keras.optimizers.Adam(
      learning_rate=learning_rate1,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07
  )

  optimizer2= tf.keras.optimizers.Adam(
      learning_rate=learning_rate2,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07
  )

  # Model Callbacks
  checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(models_output_path,'class weights', f'ResNet50_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.h5'),
    save_best_only=True) 


  base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width,3), pooling=False)


  inputs = tf.keras.Input(shape=(img_height, img_width, 3))
  x = data_augmentations(inputs)
  x = tf.keras.applications.resnet50.preprocess_input(x)
  x = base_model(x)
  pool = tf.keras.layers.GlobalAveragePooling2D()(x)
  fc1 = tf.keras.layers.Dense(1000, 'relu')(pool)
  output = tf.keras.layers.Dense(n_classes, activation='softmax')(fc1)
  model=tf.keras.Model(inputs=inputs, outputs=output)


  for layer in base_model.layers:
      layer.trainable=False


  model.compile(
    optimizer=optimizer1,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy' ])


  # In[ ]:


  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch1,
    callbacks=[checkpoint_cb, lr_cb],
    class_weight=class_weight
  )

  hist1 = pd.DataFrame(history.history)
  ## Allow fine tuning
  for layer in base_model.layers[-3:]:
      layer.trainable=True




  model.compile(
    optimizer=optimizer2,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


  # In[ ]:


  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch2,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_cb],
    class_weight=class_weight
  )
  hist2 =  pd.DataFrame(history.history)
  hist2.index = hist2.index+epoch1
  full_training_hist = pd.concat([hist1,hist2])
  # In[ ]:


  # summarize history for accuracy
  plt.subplot(1,2,1)
  plt.plot(full_training_hist['accuracy'])
  plt.plot(full_training_hist['val_accuracy'])
  plt.axvline(x = epoch1, color = 'b',linestyle='dashed', label = 'fine-tuning')
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val','fine-tuning'], loc='lower right')
  plt.subplot(1,2,2)
  # summarize history for loss
  plt.plot(full_training_hist['loss'])
  plt.plot(full_training_hist['val_loss'])
  plt.axvline(x = epoch1, color = 'b', linestyle='dashed', label = 'fine-tuning')
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val','fine-tuning'], loc='lower right')
  plt.tight_layout()
  plt.savefig(os.path.join(figures_output_path, 'graphs','class weights',f'ResNet50_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.jpg'))


  preds = model.predict(test_ds, batch_size=batch_size, verbose='auto')
  y = y = np.concatenate([y for x, y in test_ds], axis=0)
  y_hat = preds.argmax(axis=1)
  print(f"Sample output: {list(zip(y[:10], y_hat[:10]))}")

  f1 = f1_score(y, y_hat, average='weighted')

  cm = confusion_matrix(y, y_hat, labels=[0,1,2,3,4])

  print(f"F1 score: {f1}")
  print(f"Confusion Matrix: {cm}")
  f1_df = pd.DataFrame(data={"F1 score":f1}, index=[0])
  cm_df = pd.DataFrame(cm,columns = ['y_hat 0', 'y_hat 1','y_hat 2','y_hat 3','y_hat 4'])






  with pd.ExcelWriter(os.path.join(csv_output_path,'class weights',f'ResNet50_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.xlsx'),
                      mode='w') as writer:  

      f1_df.to_excel(writer, startrow=0)
      cm_df.to_excel(writer, startrow=2)
      full_training_hist.to_excel(writer, sheet_name='model history')



 



  ###########
  """
  #Xception

  """
  ##########


 

  optimizer1= tf.keras.optimizers.Adam(
      learning_rate=learning_rate1,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07
  )

  optimizer2= tf.keras.optimizers.Adam(
      learning_rate=learning_rate2,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07
  )



  # Model Callbacks
  checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(models_output_path,'class weights', f'Xception_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.h5'),
    save_best_only=True) 
  base_model = tf.keras.applications.xception.Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width,3), pooling=False)


  inputs = tf.keras.Input(shape=(img_height, img_width, 3))
  x = data_augmentations(inputs)
  x = tf.keras.applications.xception.preprocess_input(x)
  x = base_model(x)
  pool = tf.keras.layers.GlobalAveragePooling2D()(x)
  fc1 = tf.keras.layers.Dense(1000, 'relu')(pool)
  output = tf.keras.layers.Dense(n_classes, activation='softmax')(fc1)
  model=tf.keras.Model(inputs=inputs, outputs=output)




  for layer in base_model.layers:
        layer.trainable=False


  model.compile(
    optimizer=optimizer1,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy' ])


  # In[ ]:


  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch1,
    callbacks=[checkpoint_cb, lr_cb],
    class_weight=class_weight
  )

  hist1 = pd.DataFrame(history.history)
  ## Allow fine tuning
  for layer in base_model.layers[-3:]:
      layer.trainable=True




  model.compile(
    optimizer=optimizer2,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


  # In[ ]:


  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch2,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_cb],
    class_weight=class_weight
  )
  hist2 =  pd.DataFrame(history.history)
  hist2.index = hist2.index+epoch1
  full_training_hist = pd.concat([hist1,hist2])
  # In[ ]:


  # summarize history for accuracy
  plt.subplot(1,2,1)
  plt.plot(full_training_hist['accuracy'])
  plt.plot(full_training_hist['val_accuracy'])
  plt.axvline(x = epoch1, color = 'b',linestyle='dashed', label = 'fine-tuning')
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val','fine-tuning'], loc='lower right')
  plt.subplot(1,2,2)
  # summarize history for loss
  plt.plot(full_training_hist['loss'])
  plt.plot(full_training_hist['val_loss'])
  plt.axvline(x = epoch1, color = 'b', linestyle='dashed', label = 'fine-tuning')
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val','fine-tuning'], loc='lower right')
  plt.tight_layout()
  plt.savefig(os.path.join(figures_output_path, 'graphs','class weights',f'Xception_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.jpg'))


  preds = model.predict(test_ds, batch_size=batch_size, verbose='auto')
  y = y = np.concatenate([y for x, y in test_ds], axis=0)
  y_hat = preds.argmax(axis=1)
  print(f"Sample output: {list(zip(y[:10], y_hat[:10]))}")

  f1 = f1_score(y, y_hat, average='weighted')

  cm = confusion_matrix(y, y_hat, labels=[0,1,2,3,4])

  print(f"F1 score: {f1}")
  print(f"Confusion Matrix: {cm}")
  f1_df = pd.DataFrame(data={"F1 score":f1}, index=[0])
  cm_df = pd.DataFrame(cm,columns = ['y_hat 0', 'y_hat 1','y_hat 2','y_hat 3','y_hat 4'])






  with pd.ExcelWriter(os.path.join(csv_output_path,'class weights',f'Xception_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.xlsx'),
                      mode='w') as writer:  

      f1_df.to_excel(writer, startrow=0)
      cm_df.to_excel(writer, startrow=2)
      full_training_hist.to_excel(writer, sheet_name='model history')




  ###########
  """
  #MobileNet

  """
  ##########

  optimizer1= tf.keras.optimizers.Adam(
      learning_rate=learning_rate1,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07
  )

  optimizer2= tf.keras.optimizers.Adam(
      learning_rate=learning_rate2,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07
  )

  # Model Callbacks
  checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(models_output_path,'class weights', f'MobileNet_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.h5'),
    save_best_only=True) 
  base_model = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(img_height, img_width,3), pooling=False)


  inputs = tf.keras.Input(shape=(img_height, img_width, 3))
  x = data_augmentations(inputs)
  x = tf.keras.applications.mobilenet.preprocess_input(x)
  x = base_model(x)
  pool = tf.keras.layers.GlobalAveragePooling2D()(x)
  fc1 = tf.keras.layers.Dense(1024, 'relu')(pool)
  output = tf.keras.layers.Dense(n_classes, activation='softmax')(fc1)
  model=tf.keras.Model(inputs=inputs, outputs=output)




  for layer in base_model.layers:
        layer.trainable=False


  model.compile(
    optimizer=optimizer1,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy' ])


  # In[ ]:


  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch1,
    callbacks=[checkpoint_cb, lr_cb],
    class_weight=class_weight
  )

  hist1 = pd.DataFrame(history.history)
  ## Allow fine tuning
  for layer in base_model.layers[-3:]:
      layer.trainable=True




  model.compile(
    optimizer=optimizer2,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


  # In[ ]:


  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch2,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_cb],
    class_weight=class_weight
  )
  hist2 =  pd.DataFrame(history.history)
  hist2.index = hist2.index+epoch1
  full_training_hist = pd.concat([hist1,hist2])
  # In[ ]:


  # summarize history for accuracy
  plt.subplot(1,2,1)
  plt.plot(full_training_hist['accuracy'])
  plt.plot(full_training_hist['val_accuracy'])
  plt.axvline(x = epoch1, color = 'b',linestyle='dashed', label = 'fine-tuning')
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val','fine-tuning'], loc='lower right')
  plt.subplot(1,2,2)
  # summarize history for loss
  plt.plot(full_training_hist['loss'])
  plt.plot(full_training_hist['val_loss'])
  plt.axvline(x = epoch1, color = 'b', linestyle='dashed', label = 'fine-tuning')
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val','fine-tuning'], loc='lower right')
  plt.tight_layout()
  plt.savefig(os.path.join(figures_output_path, 'graphs','class weights',f'MobileNet_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.jpg'))
  plt.clf()

  preds = model.predict(test_ds, batch_size=batch_size, verbose='auto')
  y = y = np.concatenate([y for x, y in test_ds], axis=0)
  y_hat = preds.argmax(axis=1)
  print(f"Sample output: {list(zip(y[:10], y_hat[:10]))}")

  f1 = f1_score(y, y_hat, average='weighted')

  cm = confusion_matrix(y, y_hat, labels=[0,1,2,3,4])

  print(f"F1 score: {f1}")
  print(f"Confusion Matrix: {cm}")
  f1_df = pd.DataFrame(data={"F1 score":f1}, index=[0])
  cm_df = pd.DataFrame(cm,columns = ['y_hat 0', 'y_hat 1','y_hat 2','y_hat 3','y_hat 4'])






  with pd.ExcelWriter(os.path.join(csv_output_path,'class weights',f'MobileNet_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.xlsx'),
                      mode='w') as writer:  

      f1_df.to_excel(writer, startrow=0)
      cm_df.to_excel(writer, startrow=2)
      full_training_hist.to_excel(writer, sheet_name='model history')


  ###########
  """
  #DenseNet121

  """
  ##########


  optimizer1= tf.keras.optimizers.Adam(
      learning_rate=learning_rate1,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07
  )

  optimizer2= tf.keras.optimizers.Adam(
      learning_rate=learning_rate2,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07
  )

  # Model Callbacks
  checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(models_output_path, 'class weights',f'DenseNet121_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.h5'),
    save_best_only=True) 
  base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width,3), pooling=False)


  inputs = tf.keras.Input(shape=(img_height, img_width, 3))
  x = data_augmentations(inputs)
  x = tf.keras.applications.densenet.preprocess_input(x)
  x = base_model(x)
  pool = tf.keras.layers.GlobalAveragePooling2D()(x)
  output = tf.keras.layers.Dense(n_classes, activation='softmax')(pool)
  model=tf.keras.Model(inputs=inputs, outputs=output)




  for layer in base_model.layers:
        layer.trainable=False


  model.compile(
    optimizer=optimizer1,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy' ])


  # In[ ]:


  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch1,
    callbacks=[checkpoint_cb, lr_cb],
    class_weight=class_weight
  )

  hist1 = pd.DataFrame(history.history)
  ## Allow fine tuning
  for layer in base_model.layers[-3:]:
      layer.trainable=True




  model.compile(
    optimizer=optimizer2,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


  # In[ ]:


  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch2,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_cb],
    class_weight=class_weight
  )
  hist2 =  pd.DataFrame(history.history)
  hist2.index = hist2.index+epoch1
  full_training_hist = pd.concat([hist1,hist2])
  # In[ ]:


  # summarize history for accuracy
  plt.subplot(1,2,1)
  plt.plot(full_training_hist['accuracy'])
  plt.plot(full_training_hist['val_accuracy'])
  plt.axvline(x = epoch1, color = 'b',linestyle='dashed', label = 'fine-tuning')
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val','fine-tuning'], loc='lower left')
  plt.subplot(1,2,2)
  # summarize history for loss
  plt.plot(full_training_hist['loss'])
  plt.plot(full_training_hist['val_loss'])
  plt.axvline(x = epoch1, color = 'b', linestyle='dashed', label = 'fine-tuning')
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val','fine-tuning'], loc='lower left')
  plt.tight_layout()
  plt.savefig(os.path.join(figures_output_path, 'graphs','class weights',f'DenseNet121_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.jpg'))
  plt.clf()

  preds = model.predict(test_ds, batch_size=batch_size, verbose='auto')
  y = y = np.concatenate([y for x, y in test_ds], axis=0)
  y_hat = preds.argmax(axis=1)
  print(f"Sample output: {list(zip(y[:10], y_hat[:10]))}")

  f1 = f1_score(y, y_hat, average='weighted')

  cm = confusion_matrix(y, y_hat, labels=[0,1,2,3,4])

  print(f"F1 score: {f1}")
  print(f"Confusion Matrix: {cm}")
  f1_df = pd.DataFrame(data={"F1 score":f1}, index=[0])
  cm_df = pd.DataFrame(cm,columns = ['y_hat 0', 'y_hat 1','y_hat 2','y_hat 3','y_hat 4'])






  with pd.ExcelWriter(os.path.join(csv_output_path,'class weights',f'DenseNet121_{train_ds.cardinality().numpy()*batch_size/5}_{learning_rate1}.xlsx'),
                      mode='w') as writer:  

      f1_df.to_excel(writer, startrow=0)
      cm_df.to_excel(writer, startrow=2)
      full_training_hist.to_excel(writer, sheet_name='model history')
