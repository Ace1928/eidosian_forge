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
def make_seed_for_stateless_op(self):
    """Generate a new seed based on the init config.

        Note that this will not return python ints which will be frozen in the
        graph and cause stateless op to return the same value. It will only
        return value when generator is used, otherwise it will return None.

        Returns:
          A tensor with shape [2,].
        """
    self._maybe_init()
    if self._rng_type == self.RNG_STATELESS:
        return [self._seed, 0]
    elif self._rng_type == self.RNG_STATEFUL:
        return self._generator.make_seeds()[:, 0]
    return None