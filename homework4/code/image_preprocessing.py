from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import layers


def data_preprocessing(IMG_SIZE=32):
    '''
    In this function you are going to build data preprocessing layers using tf.keras
    First, resize your image to consistent shape
    Second, standardize pixel values to [0,1]
    return tf.keras.Sequential object containing the above mentioned preprocessing layers
    '''
    preprocessing_layers = tf.keras.Sequential()

    preprocessing_layers.add(tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE))

    preprocessing_layers.add(tf.keras.layers.Rescaling(scale=1./255))

    return preprocessing_layers
    

def data_augmentation():
    '''
    In this function you are going to build data augmentation layers using tf.keras
    First, add random horizontal flip
    Second, add random rotation with factor of 0.1
    Third, add random zoom (height_factor = -0.2 and width_factor = -0.2)
    return tf.keras.Sequential object containing the above mentioned augmentation layers
    '''
    data_augmentation_layers = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(height_factor=-0.2, width_factor=-0.2)
    ])

    return data_augmentation_layers


    

