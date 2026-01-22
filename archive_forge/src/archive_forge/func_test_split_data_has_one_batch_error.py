import numpy as np
import pytest
import tensorflow as tf
from autokeras.utils import data_utils
def test_split_data_has_one_batch_error():
    with pytest.raises(ValueError) as info:
        data_utils.split_dataset(tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3])).batch(12), 0.2)
    assert 'The dataset should at least contain 2 batches' in str(info.value)