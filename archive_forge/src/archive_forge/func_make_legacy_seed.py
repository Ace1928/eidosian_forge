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
def make_legacy_seed(self):
    """Create a new seed for the legacy stateful ops to use.

        When user didn't provide any original seed, this method will return
        None.  Otherwise it will increment the counter and return as the new
        seed.

        Note that it is important to generate different seed for stateful ops in
        the `tf.function`. The random ops will return same value when same seed
        is provided in the `tf.function`.

        Returns:
          int as new seed, or None.
        """
    if self._seed is not None:
        result = self._seed
        self._seed += 1
        return result
    return None