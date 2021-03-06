"""
This code is to build and train 2D U-Net
"""
import numpy as np
import sys
import subprocess
import argparse
import os
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from keras.models import Model
from keras.layers import Input, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D, add
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras import losses
import math
from keras import activations, layers
from keras.utils.conv_utils import normalize_tuple
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import csv
from utils import *
from data import load_train_data

K.set_image_data_format('channels_last')  # Tensorflow dimension ordering

# ----- paths setting -----
data_path = sys.argv[1] + "/"
model_path = data_path + "models/"
log_path = data_path + "logs/"


# ----- params for training and testing -----
batch_size = 1
cur_fold = sys.argv[2]
plane = sys.argv[3]
epoch = int(sys.argv[4])
init_lr = float(sys.argv[5])


# ----- Dice Coefficient and cost function for training -----
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return  -dice_coef(y_true, y_pred)
  

def get_unet(img_rows, img_cols, flt=64, pool_size=(2, 2, 2), init_lr=1.0e-5):
    """build and compile Neural Network"""

    print ("start building NN")
    inputs = Input((img_rows, img_cols, 1))

    conv1 = Conv2D(flt,(7,7),activation='relu', padding='same')(inputs)
    conv1= layers.BatchNormalization(epsilon=1.001e-5)(conv1)
    conv1= layers.BatchNormalization( epsilon=1.001e-5)(conv1)
    convcopy= conv1
    conv1 = Conv2D(flt, 1,activation='relu')(conv1)
    dw = layers.DepthwiseConv2D((5,5), padding='same')(conv1)
    dwd = layers.DepthwiseConv2D((7,7), padding='same', dilation_rate=3)(dw)
    pw = layers.Conv2D(flt, (1,1))(dwd)
    Lconv1=convcopy*pw
    conv1 = Conv2D(flt, 1,activation='relu')(Lconv1)
    Fconv1=convcopy+conv1
    print(Lconv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(Fconv1)

    conv2 = Conv2D(flt*2, (7,7),activation='relu', padding='same')(pool1)
    conv2= layers.BatchNormalization(epsilon=1.001e-5)(conv2)
    conv2= layers.BatchNormalization( epsilon=1.001e-5)(conv2)
    convcopy= conv2
    conv2 = Conv2D(flt*2, 1,activation='relu')(conv2)
    dw = layers.DepthwiseConv2D((5,5), padding='same')(conv2)
    dwd = layers.DepthwiseConv2D((7,7), padding='same', dilation_rate=3)(dw)
    pw = layers.Conv2D(flt*2, (1,1))(dwd)
    Lconv2=convcopy*pw
    conv2 = Conv2D(flt*2, 1,activation='relu')(Lconv2)
    Fconv2=convcopy+conv2
    print(Lconv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(Fconv2)

    conv3 = Conv2D(flt*4,(7,7),activation='relu', padding='same')(pool2)
    conv3= layers.BatchNormalization(epsilon=1.001e-5)(conv3)
    conv3= layers.BatchNormalization( epsilon=1.001e-5)(conv3)
    convcopy= conv3
    conv3 = Conv2D(flt*4, 1,activation='relu')(conv3)
    dw = layers.DepthwiseConv2D((5,5), padding='same')(conv3)
    dwd = layers.DepthwiseConv2D((7,7), padding='same', dilation_rate=3)(dw)
    pw = layers.Conv2D(flt*4, (1,1))(dwd)
    Lconv3=convcopy*pw
    conv3 = Conv2D(flt*4, 1,activation='relu')(Lconv3)
    Fconv3=convcopy+conv3
    print(Lconv2.shape)
    print(conv3.shape)
    print(Lconv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(Fconv3)

    conv4 = Conv2D(flt*8,(7,7),activation='relu', padding='same')(pool3)
    conv4= layers.BatchNormalization(epsilon=1.001e-5)(conv4)
    conv4= layers.BatchNormalization( epsilon=1.001e-5)(conv4)
    convcopy= conv4
    conv4 = Conv2D(flt*8, 1,activation='relu')(conv4)
    dw = layers.DepthwiseConv2D((5,5), padding='same')(conv4)
    dwd = layers.DepthwiseConv2D((7,7), padding='same', dilation_rate=3)(dw)
    pw = layers.Conv2D(flt*8, (1,1))(dwd)
    print("pw=", pw.shape,"   ","convcopy4=", convcopy.shape)
    Lconv4=convcopy*pw
    conv4 = Conv2D(flt*8, 1,activation='relu')(Lconv4)
    Fconv4=convcopy+conv4
    print(Lconv4.shape)
    pool4 = MaxPooling2D(pool_size=(2, 2))(Fconv4)

    conv5 = Conv2D(flt*16, (7,7), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(flt*8, (7,7), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(flt*8, (2, 2), strides=(2, 2), padding='same')(conv5), Lconv4], axis=3)
    conv6 = Conv2D(flt*8, (7, 7), activation='relu', padding='same')(up6)
    conv6 = Conv2D(flt*4, (7, 7), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(flt*4, (2, 2), strides=(2, 2), padding='same')(conv6), Lconv3], axis=3)
    conv7 = Conv2D(flt*4, (7, 7), activation='relu', padding='same')(up7)
    conv7 = Conv2D(flt*2, (7, 7), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv7), Lconv2], axis=3)
    conv8 = Conv2D(flt*2, (7, 7), activation='relu', padding='same')(up8)
    conv8 = Conv2D(flt, (7, 7), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), Lconv1], axis=3)
    conv9 = Conv2D(flt, (7, 7), activation='relu', padding='same')(up9)
    conv9 = Conv2D(flt, (7, 7), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=init_lr), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train(fold, plane, batch_size, nb_epoch,init_lr):
    """
    train an Unet model with data from load_train_data()
    Parameters
    ----------
    fold : string
        which fold is experimenting in 4-fold. It should be one of 0/1/2/3
    plane : char
        which plane is experimenting. It is from 'X'/'Y'/'Z'
    batch_size : int
        size of mini-batch
    nb_epoch : int
        number of epochs to train NN
    init_lr : float
        initial learning rate
    """

    print ("number of epoch: ", nb_epoch)
    print ("learning rate: ", init_lr)

    # --------------------- load and preprocess training data -----------------
    print ('-'*80)
    print ('         Loading and preprocessing train data...')
    print ('-'*80)

    imgs_train, imgs_mask_train = load_train_data(fold, plane)

    imgs_row = imgs_train.shape[1]
    imgs_col = imgs_train.shape[2]

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')

    # ---------------------- Create, compile, and train model ------------------------
    print ('-'*80)
    print ('		Creating and compiling model...')
    print ('-'*80)

    model = get_unet(imgs_row, imgs_col, pool_size=(2, 2, 2), init_lr=init_lr)
    print (model.summary())

    print ('-'*80)
    print ('		Fitting model...')
    print ('-'*80)

    ver = 'unet_fd%s_%s_ep%s_lr%s.csv'%(cur_fold, plane, epoch, init_lr)
    csv_logger = CSVLogger(log_path + ver)
    model_checkpoint = ModelCheckpoint(model_path + ver + ".h5",
                                       monitor='loss',
                                       save_best_only=False,
                                       period=10)

    history = model.fit(imgs_train, imgs_mask_train,
                        batch_size= batch_size, epochs= nb_epoch, verbose=1, shuffle=True,
                        callbacks=[model_checkpoint, csv_logger])


if __name__ == "__main__":

    train(cur_fold, plane, batch_size, epoch, init_lr)

    print ("training done") 
