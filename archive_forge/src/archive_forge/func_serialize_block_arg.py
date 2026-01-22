import json
import multiprocessing
import os
from typing import Optional
from typing import Tuple
import numpy as np
import tensorflow as tf
from keras_tuner.engine import hyperparameters
def serialize_block_arg(arg):
    if isinstance(arg, hyperparameters.HyperParameter):
        return hyperparameters.serialize(arg)
    return arg