from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.backend import keras
from keras_tuner.src.backend import ops
from keras_tuner.src.backend.keras import layers
from keras_tuner.src.engine import hypermodel
def sep_conv(x, num_filters, kernel_size=(3, 3), activation='relu'):
    if activation == 'selu':
        x = layers.SeparableConv2D(num_filters, kernel_size, activation='selu', padding='same', depthwise_initializer='lecun_normal', pointwise_initializer='lecun_normal')(x)
    elif activation == 'relu':
        x = layers.SeparableConv2D(num_filters, kernel_size, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    return x