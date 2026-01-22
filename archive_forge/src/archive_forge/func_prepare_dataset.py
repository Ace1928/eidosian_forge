import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import dataset_utils
from tensorflow.python.util.tf_export import keras_export
def prepare_dataset(dataset, batch_size, shuffle, seed, class_names, output_sequence_length, ragged):
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        if output_sequence_length is None and (not ragged):
            dataset = dataset.padded_batch(batch_size, padded_shapes=([None, None], []))
        else:
            dataset = dataset.batch(batch_size)
    elif shuffle:
        dataset = dataset.shuffle(buffer_size=1024, seed=seed)
    dataset.class_names = class_names
    return dataset