
"""
Created on Wed Nov 21 21:08:16 2018

@author: Chuk Yong

"""
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
#%%
# Parameters
batch_size = 32
num_classes = 10
epochs = 20
num_predictions = 20
#%%
#import
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#%%
# rescale image pixel value to between 0,1
x_train = x_train/255.
x_test = x_test/255.
#%%
# one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#%%
# Pick one of the following model and comment out the other
#model-relu
#https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
# Accuracy was 0.7489
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()
#%%
#model-elu
#Accuracy = 0.9598
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('elu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('elu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('elu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()
#%%
# Compile
# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
#%%
#W?O Augmentation
history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test)
              )
#%%
#Save model
from keras.models import load_model
# model relu
#model.save(path+'cifar10-relu.h5')  # creates a HDF5 file
#del model
#model = load_model(path+'cifar10-relu.h5')
# model elu
model.save(path+'cifar10-elu.h5')  # creates a HDF5 file
del model
model = load_model(path+'cifar10-elu.h5')


