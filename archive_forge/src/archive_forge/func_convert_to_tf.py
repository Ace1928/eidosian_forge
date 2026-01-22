import itertools
import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def convert_to_tf(x):
    if is_scipy_sparse(x):
        x = scipy_sparse_to_tf_sparse(x)
    elif is_jax_sparse(x):
        x = jax_sparse_to_tf_sparse(x)
    return x