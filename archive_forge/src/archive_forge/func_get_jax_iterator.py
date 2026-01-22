import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def get_jax_iterator(self):
    return data_adapter_utils.get_jax_iterator(self.get_numpy_iterator())