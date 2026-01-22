import numpy as np
import tensorflow as tf
from tensorflow import nest
def dataset_shape(dataset):
    return tf.compat.v1.data.get_output_shapes(dataset)