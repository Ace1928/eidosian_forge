import numpy as np
import pytest
import tensorflow as tf
from autokeras.utils import data_utils
def test_unzip_dataset_doesnt_unzip_single_dataset():
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 32, 2))
    dataset = data_utils.unzip_dataset(dataset)[0]
    dataset = data_utils.unzip_dataset(dataset)[0]
    assert data_utils.dataset_shape(dataset).as_list() == [32, 2]