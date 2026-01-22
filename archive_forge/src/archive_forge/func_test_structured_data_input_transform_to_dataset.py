import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.adapters import input_adapters
from autokeras.utils import data_utils
def test_structured_data_input_transform_to_dataset():
    x = tf.data.Dataset.from_tensor_slices(pd.read_csv(test_utils.TRAIN_CSV_PATH).to_numpy().astype(np.unicode))
    adapter = input_adapters.StructuredDataAdapter()
    x = adapter.adapt(x, batch_size=32)
    assert isinstance(x, tf.data.Dataset)