import collections
import math
import numpy as np
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import tree
def to_tensorflow_sparse_wrapper(sparse):
    from keras.src.utils.module_utils import tensorflow as tf
    row_ids = sparse.indices[:, 0]
    row_splits = tf.experimental.RowPartition.from_value_rowids(row_ids).row_splits()
    ragged_indices = tf.cast(tf.RaggedTensor.from_row_splits(sparse.indices, row_splits), tf.int64)
    ragged_values = tf.RaggedTensor.from_row_splits(sparse.values, row_splits)
    return TensorflowSparseWrapper(sparse, ragged_indices, ragged_values)