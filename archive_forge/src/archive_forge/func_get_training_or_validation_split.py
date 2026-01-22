import multiprocessing
import os
import random
import time
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def get_training_or_validation_split(samples, labels, validation_split, subset):
    """Potentially restict samples & labels to a training or validation split.

    Args:
      samples: List of elements.
      labels: List of corresponding labels.
      validation_split: Float, fraction of data to reserve for validation.
      subset: Subset of the data to return.
        Either "training", "validation", or None. If None, we return all of the
        data.

    Returns:
      tuple (samples, labels), potentially restricted to the specified subset.
    """
    if not validation_split:
        return (samples, labels)
    num_val_samples = int(validation_split * len(samples))
    if subset == 'training':
        io_utils.print_msg(f'Using {len(samples) - num_val_samples} files for training.')
        samples = samples[:-num_val_samples]
        labels = labels[:-num_val_samples]
    elif subset == 'validation':
        io_utils.print_msg(f'Using {num_val_samples} files for validation.')
        samples = samples[-num_val_samples:]
        labels = labels[-num_val_samples:]
    else:
        raise ValueError(f'`subset` must be either "training" or "validation", received: {subset}')
    return (samples, labels)