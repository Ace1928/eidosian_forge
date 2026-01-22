import collections
import math
import numpy as np
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import tree
@classmethod
def convert_to_torch_compatible(cls, x):
    return x.todense()