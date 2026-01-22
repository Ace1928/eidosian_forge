import contextlib
import copy
import gc
import math
import os
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import errors
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.backend import config
from keras_tuner.src.backend import keras
from keras_tuner.src.engine import base_tuner
from keras_tuner.src.engine import tuner_utils
def maybe_compute_model_size(model):
    """Compute the size of a given model, if it has been built."""
    if model.built:
        params = [math.prod(p.shape) for p in model.trainable_weights]
        return int(np.sum(params))
    return 0