import numpy as np
import tensorflow as tf
from autokeras import preprocessors
from autokeras.preprocessors import encoders
from autokeras.utils import data_utils
def test_one_hot_encoder_decode_to_same_string():
    encoder = encoders.OneHotEncoder(['a', 'b', 'c'])
    result = encoder.postprocess(np.eye(3))
    assert np.array_equal(result, np.array([['a'], ['b'], ['c']]))