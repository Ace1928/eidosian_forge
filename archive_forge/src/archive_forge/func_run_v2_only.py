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
def run_v2_only(func=None):
    """Execute the decorated test only if running in v2 mode.

  This function is intended to be applied to tests that exercise v2 only
  functionality. If the test is run in v1 mode it will simply be skipped.

  See go/tf-test-decorator-cheatsheet for the decorators to use in different
  v1/v2/eager/graph combinations.

  Args:
    func: function to be annotated. If `func` is None, this method returns a
      decorator the can be applied to a function. If `func` is not None this
      returns the decorator applied to `func`.

  Returns:
    Returns a decorator that will conditionally skip the decorated test method.
  """

    def decorator(f):
        if tf_inspect.isclass(f):
            raise ValueError('`run_v2_only` only supports test methods.')

        def decorated(self, *args, **kwargs):
            if not tf2.enabled():
                self.skipTest('Test is only compatible with v2')
            return f(self, *args, **kwargs)
        return decorated
    if func is not None:
        return decorator(func)
    return decorator