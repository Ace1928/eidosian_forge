import os
import warnings
from functools import partial
from math import ceil
from uuid import uuid4
import numpy as np
import pyarrow as pa
from multiprocess import get_context
from .. import config
def dataset_to_tf(dataset, cols_to_retain, collate_fn, collate_fn_args, columns_to_np_types, output_signature, shuffle, batch_size, drop_remainder):
    """Create a tf.data.Dataset from the underlying Dataset. This is a single-process method - the multiprocess
    equivalent is multiprocess_dataset_to_tf.

    Args:
        dataset (`Dataset`): Dataset to wrap with tf.data.Dataset.
        cols_to_retain (`List[str]`): Dataset column(s) to load in the
            tf.data.Dataset. It is acceptable to include column names that are created by the `collate_fn` and
            that do not exist in the original dataset.
        collate_fn(`Callable`): A function or callable object (such as a `DataCollator`) that will collate
            lists of samples into a batch.
        collate_fn_args (`Dict`): A  `dict` of keyword arguments to be passed to the
            `collate_fn`. Can be empty.
        columns_to_np_types (`Dict[str, np.dtype]`): A `dict` mapping column names to numpy dtypes.
        output_signature (`Dict[str, tf.TensorSpec]`): A `dict` mapping column names to
            `tf.TensorSpec` objects.
        shuffle(`bool`): Shuffle the dataset order when loading. Recommended True for training, False for
            validation/evaluation.
        batch_size (`int`, default `None`): Size of batches to load from the dataset. Defaults to `None`, which implies that
            the dataset won't be batched, but the returned dataset can be batched later with `tf_dataset.batch(batch_size)`.
        drop_remainder(`bool`, default `None`): Drop the last incomplete batch when loading. If not provided,
            defaults to the same setting as shuffle.

    Returns:
        `tf.data.Dataset`
    """
    if config.TF_AVAILABLE:
        import tensorflow as tf
    else:
        raise ImportError('Called a Tensorflow-specific function but Tensorflow is not installed.')
    if hasattr(tf, 'random_index_shuffle'):
        random_index_shuffle = tf.random_index_shuffle
    elif hasattr(tf.random.experimental, 'index_shuffle'):
        random_index_shuffle = tf.random.experimental.index_shuffle
    else:
        if len(dataset) > 10000000:
            warnings.warn('to_tf_dataset() can be memory-inefficient on versions of TensorFlow older than 2.9. If you are iterating over a dataset with a very large number of samples, consider upgrading to TF >= 2.9.')
        random_index_shuffle = None
    getter_fn = partial(np_get_batch, dataset=dataset, cols_to_retain=cols_to_retain, collate_fn=collate_fn, collate_fn_args=collate_fn_args, columns_to_np_types=columns_to_np_types, return_dict=False)
    tout = [tf.dtypes.as_dtype(dtype) for dtype in columns_to_np_types.values()]

    @tf.function(input_signature=[tf.TensorSpec(None, tf.int64)])
    def fetch_function(indices):
        output = tf.py_function(getter_fn, inp=[indices], Tout=tout)
        return {key: output[i] for i, key in enumerate(columns_to_np_types.keys())}
    tf_dataset = tf.data.Dataset.range(len(dataset))
    if shuffle and random_index_shuffle is not None:
        base_seed = tf.fill((3,), value=tf.cast(-1, dtype=tf.int64))

        def scan_random_index(state, index):
            if tf.reduce_all(state == -1):
                state = tf.random.uniform(shape=(3,), maxval=2 ** 62, dtype=tf.int64)
            shuffled_index = random_index_shuffle(index=index, seed=state, max_index=len(dataset) - 1)
            return (state, shuffled_index)
        tf_dataset = tf_dataset.scan(base_seed, scan_random_index)
    elif shuffle:
        tf_dataset = tf_dataset.shuffle(tf_dataset.cardinality())
    if batch_size is not None:
        tf_dataset = tf_dataset.batch(batch_size, drop_remainder=drop_remainder)
    tf_dataset = tf_dataset.map(fetch_function)
    if batch_size is not None:

        def ensure_shapes(input_dict):
            return {key: tf.ensure_shape(val, output_signature[key].shape) for key, val in input_dict.items()}
    else:

        def ensure_shapes(input_dict):
            return {key: tf.ensure_shape(val, output_signature[key].shape[1:]) for key, val in input_dict.items()}
    return tf_dataset.map(ensure_shapes)