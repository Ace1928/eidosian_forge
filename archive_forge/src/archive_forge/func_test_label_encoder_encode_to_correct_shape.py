import numpy as np
import tensorflow as tf
from autokeras import preprocessors
from autokeras.preprocessors import encoders
from autokeras.utils import data_utils
def test_label_encoder_encode_to_correct_shape():
    encoder = encoders.LabelEncoder(['a', 'b'])
    dataset = tf.data.Dataset.from_tensor_slices([['a'], ['b']]).batch(32)
    result = encoder.transform(dataset)
    assert data_utils.dataset_shape(result).as_list() == [None, 1]