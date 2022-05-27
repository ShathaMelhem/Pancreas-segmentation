"""
This code is to build and train 2D U-Net
"""
import numpy as np
import sys
import subprocess
import argparse
import os

from keras.models import Model
from keras.layers import Input, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D, add
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras import losses

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
  
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, \
    Add, Activation, \
    BatchNormalization
from tensorflow.python.keras.models import Model





def residual_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    input_tensor = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    res_tensor = Add()([input_tensor, x])
    res_tensor = Activation('relu')(res_tensor)
    return res_tensor


def dilated_center_block(input_tensor, num_filters):

    dilation_1 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(1, 1), padding='same')(input_tensor)
    dilation_1 = Activation('relu')(dilation_1)

    dilation_2 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(2, 2), padding='same')(dilation_1)
    dilation_2 = Activation('relu')(dilation_2)

    dilation_4 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(4, 4), padding='same')(dilation_2)
    dilation_4 = Activation('relu')(dilation_4)

    dilation_8 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(8, 8), padding='same')(dilation_4)
    dilation_8 = Activation('relu')(dilation_8)

    final_diliation = Add()([input_tensor, dilation_1, dilation_2, dilation_4, dilation_8])

    return final_diliation


def decoder_block(input_tensor, num_filters):
    decoder_tensor = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
    decoder_tensor = BatchNormalization()(decoder_tensor)
    decoder_tensor = Activation('relu')(decoder_tensor)

    decoder_tensor = Conv2DTranspose(num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(decoder_tensor)
    decoder_tensor = BatchNormalization()(decoder_tensor)
    decoder_tensor = Activation('relu')(decoder_tensor)

    decoder_tensor = Conv2D(num_filters, (1, 1), padding='same')(decoder_tensor)
    decoder_tensor = BatchNormalization()(decoder_tensor)
    decoder_tensor = Activation('relu')(decoder_tensor)
    return decoder_tensor


def encoder_block(input_tensor, num_filters, num_res_blocks):
    encoded = residual_block(input_tensor, num_filters)
    while num_res_blocks > 1:
        encoded = residual_block(encoded, num_filters)
        num_res_blocks -= 1
    encoded_pool = MaxPooling2D((2, 2), strides=(2, 2))(encoded)
    return encoded, encoded_pool


def create_dlinknet(img_rows, img_cols, flt=64, pool_size=(2, 2, 2), init_lr=1.0e-5):
    inputs = Input((img_rows, img_cols, 1))
    inputs_ = Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
    inputs_ = BatchNormalization()(inputs_)
    inputs_ = Activation('relu')(inputs_)
    max_pool_inputs = MaxPooling2D((2, 2), strides=(2, 2))(inputs_)

    encoded_1, encoded_pool_1 = encoder_block(max_pool_inputs, num_filters=64, num_res_blocks=3)
    encoded_2, encoded_pool_2 = encoder_block(encoded_pool_1, num_filters=128, num_res_blocks=4)
    encoded_3, encoded_pool_3 = encoder_block(encoded_pool_2, num_filters=256, num_res_blocks=6)
    encoded_4, encoded_pool_4 = encoder_block(encoded_pool_3, num_filters=512, num_res_blocks=3)

    center = dilated_center_block(encoded_4, 512)

    decoded_1 = Add()([decoder_block(center, 256), encoded_3])
    decoded_2 = Add()([decoder_block(decoded_1, 128), encoded_2])
    decoded_3 = Add()([decoder_block(decoded_2, 64), encoded_1])
    decoded_4 = decoder_block(decoded_3, 64)

    final = Conv2DTranspose(32, kernel_size=(3, 3), padding='same')(decoded_4)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(final)
    model_i = Model(inputs=[inputs], outputs=[outputs])
    model_i.compile(optimizer='adam', loss=combined_loss, metrics=[dice_coeff])
    model_i.summary()
    # model_i.load_weights(save_model_path)
    return model_i


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

    model = create_dlinknet(imgs_row, imgs_col, pool_size=(2, 2, 2), init_lr=init_lr)
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

model = create_dlinknet()
