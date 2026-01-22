import keras_tuner
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
import autokeras as ak
from autokeras import hyper_preprocessors
from autokeras import nodes as input_module
from autokeras import preprocessors
from autokeras import test_utils
from autokeras.blocks import heads as head_module
def test_multi_label_loss():
    head = head_module.ClassificationHead(name='a', multi_label=True, num_classes=8, shape=(8,))
    input_node = keras.Input(shape=(5,))
    output_node = head.build(keras_tuner.HyperParameters(), input_node)
    model = keras.Model(input_node, output_node)
    assert model.layers[-1].activation.__name__ == 'sigmoid'
    assert head.loss.name == 'binary_crossentropy'