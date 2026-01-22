import collections
import enum
import functools
import json
import numpy as np
import tensorflow.compat.v2 as tf
import wrapt
from keras.src.saving import serialization_lib
from keras.src.saving.legacy import serialization
from keras.src.saving.legacy.saved_model.utils import in_tf_saved_model_scope
from tensorflow.python.framework import type_spec_registry
def decode_and_deserialize(json_string, module_objects=None, custom_objects=None):
    """Decodes the JSON and deserializes any Keras objects found in the dict."""
    return json.loads(json_string, object_hook=functools.partial(_decode_helper, deserialize=True, module_objects=module_objects, custom_objects=custom_objects))