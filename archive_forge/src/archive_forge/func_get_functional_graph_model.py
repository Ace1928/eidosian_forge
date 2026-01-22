import keras.src as keras
from keras.src.testing_infra import test_utils
@staticmethod
def get_functional_graph_model(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(32, activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(num_classes)(x)
    return keras.Model(inputs, outputs)