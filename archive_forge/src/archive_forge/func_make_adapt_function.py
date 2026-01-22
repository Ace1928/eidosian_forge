import abc
import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as trackable
def make_adapt_function(self):
    """Creates a function to execute one step of `adapt`.

    This method can be overridden to support custom adapt logic.
    This method is called by `PreprocessingLayer.adapt`.

    Typically, this method directly controls `tf.function` settings,
    and delegates the actual state update logic to
    `PreprocessingLayer.update_state`.

    This function is cached the first time `PreprocessingLayer.adapt`
    is called. The cache is cleared whenever `PreprocessingLayer.compile`
    is called.

    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, retrieve a batch, and update the state of the
      layer.
    """
    if self._adapt_function is not None:
        return self._adapt_function

    def adapt_step(iterator):
        data = next(iterator)
        self._adapt_maybe_build(data)
        self.update_state(data)
    if self._steps_per_execution.numpy().item() == 1:
        adapt_fn = adapt_step
    else:

        def adapt_fn(iterator):
            for _ in math_ops.range(self._steps_per_execution):
                adapt_step(iterator)
    if not self._run_eagerly:
        adapt_fn = def_function.function(adapt_fn)
    self._adapt_function = adapt_fn
    return self._adapt_function