import importlib
import inspect
import threading
import types
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.saving import object_registration
from keras.src.saving.legacy import serialization as legacy_serialization
from keras.src.saving.legacy.saved_model.utils import in_tf_saved_model_scope
from keras.src.utils import generic_utils
from tensorflow.python.util import tf_export
from tensorflow.python.util.tf_export import keras_export
def record_object_after_serialization(obj, config):
    """Call after serializing an object, to keep track of its config."""
    if config['module'] == '__main__':
        config['module'] = None
    if not getattr(SHARED_OBJECTS, 'enabled', False):
        return
    obj_id = int(id(obj))
    if obj_id not in SHARED_OBJECTS.id_to_config_map:
        SHARED_OBJECTS.id_to_config_map[obj_id] = config
    else:
        config['shared_object_id'] = obj_id
        prev_config = SHARED_OBJECTS.id_to_config_map[obj_id]
        prev_config['shared_object_id'] = obj_id