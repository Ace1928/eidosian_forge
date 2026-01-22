from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
from absl import logging
import six
import tensorflow as tf
Returns an optimizer_v2.OptimizerV2 instance.

  Supports the following types for the given `opt`:
  * An `optimizer_v2.OptimizerV2` instance: Returns the given `opt`.
  * A string: Creates an `optimizer_v2.OptimizerV2` subclass with the given
  `learning_rate`.
    Supported strings:
    * 'Adagrad': Returns an tf.keras.optimizers.Adagrad.
    * 'Adam': Returns an tf.keras.optimizers.Adam.
    * 'Ftrl': Returns an tf.keras.optimizers.Ftrl.
    * 'RMSProp': Returns an tf.keras.optimizers.RMSProp.
    * 'SGD': Returns a tf.keras.optimizers.SGD.

  Args:
    opt: An `tf.keras.optimizers.Optimizer` instance, or string, as discussed
      above.
    learning_rate: A float. Only used if `opt` is a string. If None, and opt is
      string, it will use the default learning_rate of the optimizer.

  Returns:
    An `tf.keras.optimizers.Optimizer` instance.

  Raises:
    ValueError: If `opt` is an unsupported string.
    ValueError: If `opt` is a supported string but `learning_rate` was not
      specified.
    ValueError: If `opt` is none of the above types.
  