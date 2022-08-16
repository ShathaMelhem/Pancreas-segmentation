"""
This code is to build and train 2D Mathitís U-Net
"""
import numpy as np
import sys
import subprocess
import argparse
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from keras.models import Model
from keras.layers import *
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
###########################################################################
def get_unet(img_rows, img_cols, flt=64, pool_size=(2, 2, 2), init_lr=1.0e-5):
    """build and compile Neural Network"""

    print ("start building NN")
    inputs = Input((img_rows, img_cols, 1))

    convMain =Conv2D(flt, (7, 7), activation='relu', padding='same')(inputs)
    conv1p1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(convMain)
    drop1p1= Dropout(0.3)(conv1p1)
    conv1p1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(drop1p1)

    conv1p2= Conv2D(flt, (3, 3), activation='relu', padding='same')(convMain)
    drop1p2= Dropout(0.2)(conv1p2)
    conv1p2 =Conv2D(flt, (3, 3), activation='relu', padding='same')(drop1p2)

    Fadd1= Add()([convMain, conv1p1,conv1p2])
    pool1 = MaxPooling2D(pool_size=(2, 2))(Fadd1)

#######################################################################
    copy_2=Conv2D(flt*2, (1, 1), activation='relu', padding='same') (pool1)
    copy2=copy_2
    print(copy2.shape)
    conv2p1 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(copy_2)
    drop2p1= Dropout(0.3)(conv2p1)
    conv2p1 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(drop2p1)
    print(conv2p1.shape)

    conv2p2= Conv2D(flt*2, (3, 3), activation='relu', padding='same')(copy_2)
    drop2p2= Dropout(0.2)(conv2p2)
    conv2p2 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(drop2p2)

    Fadd2= Add()([copy2, conv2p1,conv2p2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(Fadd2)
####################################################################
    copy_3=Conv2D(flt*4, (1, 1), activation='relu', padding='same') (pool2)
    copy3=copy_3
    print(copy3.shape)
    conv3p1 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(copy_3)
    drop3p1= Dropout(0.3)(conv3p1)
    conv3p1 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(drop3p1)

    conv3p2= Conv2D(flt*4, (3, 3), activation='relu', padding='same')(copy_3)
    drop3p2= Dropout(0.2)(conv3p2)
    conv3p2 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(drop3p2)

    Fadd3= Add()([copy3, conv3p1,conv3p2])
    pool3 = MaxPooling2D(pool_size=(2, 2))(Fadd3)
######################################################################
    copy_4=Conv2D(flt*8, (1, 1), activation='relu', padding='same') (pool3)
    copy4=copy_4
    conv4p1 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(pool3)
    drop4p1= Dropout(0.3)(conv4p1)
    conv4p1 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(drop4p1)

    conv4p2=Conv2D(flt*8, (3, 3), activation='relu', padding='same')(pool3)
    drop4p2= Dropout(0.2)(conv4p2)
    conv4p2 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(drop4p2)

    Fadd4= Add()([copy4, conv4p1,conv4p2])
    pool4 = MaxPooling2D(pool_size=(2, 2))(Fadd4)
#*************************************************************************

    conv5 = Conv2D(flt*16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(conv5)
#*******************************************************************************

    up6 = concatenate([Conv2DTranspose(flt*8, (2, 2), strides=(2, 2), padding='same')(conv5), Fadd4], axis=3)
    #conv6 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(up6)
    copy_6=Conv2D(flt*8, (1, 1), activation='relu', padding='same') (up6)
    copy6=copy_6
    conv6p1 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(copy_6)
    drop6p1= Dropout(0.3)(conv6p1)
    conv6p1 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(drop6p1)

    conv6p2= Conv2D(flt*8, (3, 3), activation='relu', padding='same')(copy_6)
    drop6p2= Dropout(0.2)(conv6p2)
    conv6p2 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(drop6p2)

    Fadd6= Add()([copy6, conv6p1,conv6p2])
    conv6 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(Fadd6)
###############################################################################################################
    up7 = concatenate([Conv2DTranspose(flt*4, (2, 2), strides=(2, 2), padding='same')(conv6), Fadd3], axis=3)
    #conv7 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(up7)
    copy_7=Conv2D(flt*4, (1, 1), activation='relu', padding='same') (up7)
    copy7=copy_7
    conv7p1 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(copy_7)
    drop7p1= Dropout(0.3)(conv7p1)
    conv7p1 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(drop7p1)

    conv7p2= Conv2D(flt*4, (3, 3), activation='relu', padding='same')(copy_7)
    drop7p2= Dropout(0.2)(conv7p2)
    conv7p2 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(drop7p2)

    Fadd7= Add()([copy7, conv7p1,conv7p2])
    conv7 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(Fadd7)
##############################################################################################################
    up8 = concatenate([Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv7), Fadd2], axis=3)
    #conv8 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(up8)
    copy_8=Conv2D(flt*2, (1, 1), activation='relu', padding='same') (up8)
    copy8=copy_8
    conv8p1 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(copy_8)
    drop8p1= Dropout(0.3)(conv8p1)
    conv8p1 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(drop8p1)

    conv8p2= Conv2D(flt*2, (3, 3), activation='relu', padding='same')(copy_8)
    drop8p2= Dropout(0.2)(conv8p2)
    conv8p2 =Conv2D(flt*2, (3, 3), activation='relu', padding='same')(drop8p2)

    Fadd8= Add()([copy8, conv8p1,conv8p2])
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same')(Fadd8)
################################################################################################################
    up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), Fadd1], axis=3)
    #conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same')(up9)
    copy_9=Conv2D(flt, (1, 1), activation='relu', padding='same') (up9)
    copy9=copy_9
    conv9p1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(copy_9)
    drop9p1= Dropout(0.1)(conv9p1)
    conv9p1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(drop9p1)

    conv9p2= Conv2D(flt, (3, 3), activation='relu', padding='same')(copy_9)
    drop9p2= Dropout(0.2)(conv9p2)
    conv9p2 = Conv2D(flt, (3, 3), activation='relu', padding='same')(drop9p2)

    Fadd9= Add()([copy9, conv9p1,conv9p2])
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same')(Fadd9)
##########################################################################################################
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

    ver = 'unetfinalF_fd%s_%s_ep%s_lr%s.csv'%(cur_fold, plane, epoch, init_lr)
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
