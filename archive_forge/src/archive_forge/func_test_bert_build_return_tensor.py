import keras_tuner
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_bert_build_return_tensor():
    block = blocks.BertBlock()
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(1,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1