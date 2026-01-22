import binascii
import codecs
import importlib
import marshal
import os
import re
import sys
import threading
import time
import types as python_types
import warnings
import weakref
import numpy as np
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
def serialize_keras_class_and_config(cls_name, cls_config, obj=None, shared_object_id=None):
    """Returns the serialization of the class with the given config."""
    base_config = {'class_name': cls_name, 'config': cls_config}
    if shared_object_id is not None:
        base_config[SHARED_OBJECT_KEY] = shared_object_id
    if _shared_object_saving_scope() is not None and obj is not None:
        shared_object_config = _shared_object_saving_scope().get_config(obj)
        if shared_object_config is None:
            return _shared_object_saving_scope().create_config(base_config, obj)
        return shared_object_config
    return base_config