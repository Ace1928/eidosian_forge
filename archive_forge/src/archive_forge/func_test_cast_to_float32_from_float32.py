import numpy as np
import pytest
import tensorflow as tf
from autokeras.utils import data_utils
def test_cast_to_float32_from_float32():
    tensor = tf.constant([0.1, 0.2], dtype=tf.float32)
    assert tf.float32 == data_utils.cast_to_float32(tensor).dtype