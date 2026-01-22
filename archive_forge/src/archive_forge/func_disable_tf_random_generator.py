import collections
import itertools
import json
import os
import random
import sys
import threading
import warnings
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend_config
from keras.src.distribute import distribute_coordinator_utils as dc
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.engine import keras_tensor
from keras.src.utils import control_flow_util
from keras.src.utils import object_identity
from keras.src.utils import tf_contextlib
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager.context import get_config
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
@keras_export('keras.backend.experimental.disable_tf_random_generator', v1=[])
def disable_tf_random_generator():
    """Disable the `tf.random.Generator` as the RNG for Keras.

    See `tf.keras.backend.experimental.is_tf_random_generator_enabled` for more
    details.
    """
    global _USE_GENERATOR_FOR_RNG
    _USE_GENERATOR_FOR_RNG = False