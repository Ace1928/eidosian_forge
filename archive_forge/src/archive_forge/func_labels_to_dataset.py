import multiprocessing
import os
import random
import time
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def labels_to_dataset(labels, label_mode, num_classes):
    """Create a tf.data.Dataset from the list/tuple of labels.

    Args:
      labels: list/tuple of labels to be converted into a tf.data.Dataset.
      label_mode: String describing the encoding of `labels`. Options are:
      - 'binary' indicates that the labels (there can be only 2) are encoded as
        `float32` scalars with values 0 or 1 (e.g. for `binary_crossentropy`).
      - 'categorical' means that the labels are mapped into a categorical
        vector.  (e.g. for `categorical_crossentropy` loss).
      num_classes: number of classes of labels.

    Returns:
      A `Dataset` instance.
    """
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    if label_mode == 'binary':
        label_ds = label_ds.map(lambda x: tf.expand_dims(tf.cast(x, 'float32'), axis=-1), num_parallel_calls=tf.data.AUTOTUNE)
    elif label_mode == 'categorical':
        label_ds = label_ds.map(lambda x: tf.one_hot(x, num_classes), num_parallel_calls=tf.data.AUTOTUNE)
    return label_ds