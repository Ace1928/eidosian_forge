import collections
import contextlib
import functools
import itertools
import threading
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import tf_decorator
def get_v2_optimizer(name, **kwargs):
    """Get the v2 optimizer requested.

  This is only necessary until v2 are the default, as we are testing in Eager,
  and Eager + v1 optimizers fail tests. When we are in v2, the strings alone
  should be sufficient, and this mapping can theoretically be removed.

  Args:
    name: string name of Keras v2 optimizer.
    **kwargs: any kwargs to pass to the optimizer constructor.

  Returns:
    Initialized Keras v2 optimizer.

  Raises:
    ValueError: if an unknown name was passed.
  """
    try:
        return _V2_OPTIMIZER_MAP[name](**kwargs)
    except KeyError:
        raise ValueError('Could not find requested v2 optimizer: {}\nValid choices: {}'.format(name, list(_V2_OPTIMIZER_MAP.keys())))