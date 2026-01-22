import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from autokeras import keras_layers as layer_module
def test_call_multi_with_single_column_return_right_shape():
    x_train = np.array([['a'], ['b'], ['a']])
    layer = layer_module.MultiCategoryEncoding(encoding=[layer_module.INT])
    layer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(32))
    assert layer(x_train).shape == (3, 1)