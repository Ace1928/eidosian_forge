import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from autokeras import keras_layers as layer_module
def test_cast_to_float32_return_float32_tensor(tmp_path):
    layer = layer_module.CastToFloat32()
    tensor = layer(tf.constant(['0.3'], dtype=tf.string))
    assert tf.float32 == tensor.dtype