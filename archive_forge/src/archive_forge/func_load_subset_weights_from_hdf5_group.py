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
def load_subset_weights_from_hdf5_group(f):
    """Load layer weights of a model from hdf5.

    Args:
        f: A pointer to a HDF5 group.

    Returns:
        List of NumPy arrays of the weight values.

    Raises:
        ValueError: in case of mismatch between provided model
            and weights file.
    """
    weight_names = load_attributes_from_hdf5_group(f, 'weight_names')
    return [np.asarray(f[weight_name]) for weight_name in weight_names]