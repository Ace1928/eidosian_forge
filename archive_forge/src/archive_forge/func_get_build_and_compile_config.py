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
def get_build_and_compile_config(obj, config):
    if hasattr(obj, 'get_build_config'):
        build_config = obj.get_build_config()
        if build_config is not None:
            config['build_config'] = serialize_dict(build_config)
    if hasattr(obj, 'get_compile_config'):
        compile_config = obj.get_compile_config()
        if compile_config is not None:
            config['compile_config'] = serialize_dict(compile_config)
    return