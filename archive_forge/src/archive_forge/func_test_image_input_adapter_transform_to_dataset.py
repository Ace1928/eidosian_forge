import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.adapters import input_adapters
from autokeras.utils import data_utils
def test_image_input_adapter_transform_to_dataset():
    x = test_utils.generate_data()
    adapter = input_adapters.ImageAdapter()
    assert isinstance(adapter.adapt(x, batch_size=32), tf.data.Dataset)