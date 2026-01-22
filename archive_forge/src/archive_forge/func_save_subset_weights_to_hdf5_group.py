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
def save_subset_weights_to_hdf5_group(f, weights):
    """Save top-level weights of a model to a HDF5 group.

    Args:
        f: HDF5 group.
        weights: List of weight variables.
    """
    weight_values = [backend.convert_to_numpy(w) for w in weights]
    weight_names = [w.name.encode('utf8') for w in weights]
    save_attributes_to_hdf5_group(f, 'weight_names', weight_names)
    for name, val in zip(weight_names, weight_values):
        param_dset = f.create_dataset(name, val.shape, dtype=val.dtype)
        if not val.shape:
            param_dset[()] = val
        else:
            param_dset[:] = val