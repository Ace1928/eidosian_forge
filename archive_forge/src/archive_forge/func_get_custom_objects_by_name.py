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
def get_custom_objects_by_name(item, custom_objects=None):
    """Returns the item if it is in either local or global custom objects."""
    if item in _GLOBAL_CUSTOM_OBJECTS:
        return _GLOBAL_CUSTOM_OBJECTS[item]
    elif custom_objects and item in custom_objects:
        return custom_objects[item]
    return None