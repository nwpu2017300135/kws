import argparse
import os
import sys
import tensorflow as tf
import input_data
import matplotlib.pyplot as plt
import numpy as np
import math
import keras
import util
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
def trans(dataset,batch_size):
    while 1:
        tmp=dataset.next_batch(batch_size)
        xx=np.array(tmp[0])
        yy=np.array(tmp[1])
        yy = keras.utils.to_categorical(yy,3)
#        print(xx.shape)
 #       print(yy.shape)
        yield xx,yy
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
batch_size =64
num_classes = 3
epochs =350
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_kws_trained_model.h5'

train,test = input_data.read_data_sets()
(xxx,yyy)=(test._exampls,test._labels)
(x_test,y_test)=(np.array(xxx),np.array(yyy))
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
#y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()
model.add(Dense(128,input_shape=(1640,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.SGD(lr=0.001, momentum=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
model.fit_generator(trans(train,batch_size),
              epochs=epochs,
              validation_data=(x_test, y_test),
              steps_per_epoch=13,
              verbose=2,
              workers=0)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
# process in paper
probability=model.predict(x_test,verbose=0,batch_size=batch_size)
confidence=util.posteriorHandling(probability,test.fbank_end_frame)
#print(confidence)
l1,l2=util.do_eval(confidence)
l1=np.array(l1)
l2=np.array(l2)
np.save("l5",l1)
np.save("l6",l2)
