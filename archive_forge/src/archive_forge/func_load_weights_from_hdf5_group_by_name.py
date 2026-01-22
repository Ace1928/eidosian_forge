import json
import os
import numpy as np
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.saving import model_config as model_config_lib
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
def load_weights_from_hdf5_group_by_name(f, layers, skip_mismatch=False):
    """Implements name-based weight loading.

  (instead of topological weight loading).

  Layers that have no matching name are skipped.

  Args:
      f: A pointer to a HDF5 group.
      layers: a list of target layers.
      skip_mismatch: Boolean, whether to skip loading of layers
          where there is a mismatch in the number of weights,
          or a mismatch in the shape of the weights.

  Raises:
      ValueError: in case of mismatch between provided layers
          and weights file and skip_match=False.
  """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version']
        if hasattr(original_keras_version, 'decode'):
            original_keras_version = original_keras_version.decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend']
        if hasattr(original_backend, 'decode'):
            original_backend = original_backend.decode('utf8')
    else:
        original_backend = None
    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
    index = {}
    for layer in layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
        for layer in index.get(name, []):
            symbolic_weights = _legacy_weights(layer)
            weight_values = preprocess_weights_for_loading(layer, weight_values, original_keras_version, original_backend)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    logging.warning('Skipping loading of weights for layer {}'.format(layer.name) + ' due to mismatch in number of weights ({} vs {}).'.format(len(symbolic_weights), len(weight_values)))
                    continue
                raise ValueError('Layer #' + str(k) + ' (named "' + layer.name + '") expects ' + str(len(symbolic_weights)) + ' weight(s), but the saved weights' + ' have ' + str(len(weight_values)) + ' element(s).')
            for i in range(len(weight_values)):
                if backend.int_shape(symbolic_weights[i]) != weight_values[i].shape:
                    if skip_mismatch:
                        logging.warning('Skipping loading of weights for layer {}'.format(layer.name) + ' due to mismatch in shape ({} vs {}).'.format(symbolic_weights[i].shape, weight_values[i].shape))
                        continue
                    raise ValueError('Layer #' + str(k) + ' (named "' + layer.name + '"), weight ' + str(symbolic_weights[i]) + ' has shape {}'.format(backend.int_shape(symbolic_weights[i])) + ', but the saved weight has shape ' + str(weight_values[i].shape) + '.')
                else:
                    weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
    backend.batch_set_value(weight_value_tuples)