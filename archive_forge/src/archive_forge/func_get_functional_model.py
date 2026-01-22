import collections
import keras.src as keras
def get_functional_model():
    inputs = keras.Input(shape=(4,))
    x = keras.layers.Dense(4, activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(2)(x)
    return keras.Model(inputs, outputs)