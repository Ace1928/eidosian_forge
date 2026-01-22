import keras_tuner
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_rnn_build_return_tensor():
    block = blocks.RNNBlock(bidirectional=False)
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 10), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1