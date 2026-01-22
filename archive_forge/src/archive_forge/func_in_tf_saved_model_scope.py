import copy
import itertools
import threading
import types
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer_utils
from keras.src.utils import control_flow_util
from keras.src.utils import tf_contextlib
from keras.src.utils.generic_utils import LazyLoader
from keras.src.utils.layer_utils import CallFunctionSpec
def in_tf_saved_model_scope():
    return _save_options_context.in_tf_saved_model_scope