import itertools
import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def peek_and_restore(generator):
    element = next(generator)
    return (element, itertools.chain([element], generator))