import json
import os
import warnings
import numpy as np
from absl import logging
from keras.src import backend
from keras.src import optimizers
from keras.src.backend.common import global_state
from keras.src.legacy.saving import json_utils
from keras.src.legacy.saving import saving_options
from keras.src.legacy.saving import saving_utils
from keras.src.saving import object_registration
from keras.src.utils import io_utils
Legacy weight order converter.

    For legacy reason, the layer.weights was in the order of
    [self.trainable_weights + self.non_trainable_weights], and this order was
    used for preserving the weights in h5 format. The new order of layer.weights
    are the same as layer.get_weights() which is more intuitive for user. To
    keep supporting the existing saved h5 file, this method should be used to
    save/load weights.

    Args:
        layer: a `Model` or `Layer` instance.

    Returns:
        A list of variables with the legacy weight order.
    