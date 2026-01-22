import itertools
import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def tf_sparse_to_jax_sparse(x):
    import jax.experimental.sparse as jax_sparse
    from keras.src.backend.tensorflow.core import convert_to_numpy
    values = convert_to_numpy(x.values)
    indices = convert_to_numpy(x.indices)
    return jax_sparse.BCOO((values, indices), shape=x.shape)