from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
from absl import logging
import six
import tensorflow as tf
def get_optimizer_instance(opt, learning_rate=None):
    """Returns an optimizer instance.

  Supports the following types for the given `opt`:
  * An `Optimizer` instance: Returns the given `opt`.
  * A string: Creates an `Optimizer` subclass with the given `learning_rate`.
    Supported strings:
    * 'Adagrad': Returns an `AdagradOptimizer`.
    * 'Adam': Returns an `AdamOptimizer`.
    * 'Ftrl': Returns an `FtrlOptimizer`.
    * 'RMSProp': Returns an `RMSPropOptimizer`.
    * 'SGD': Returns a `GradientDescentOptimizer`.

  Args:
    opt: An `Optimizer` instance, or string, as discussed above.
    learning_rate: A float. Only used if `opt` is a string.

  Returns:
    An `Optimizer` instance.

  Raises:
    ValueError: If `opt` is an unsupported string.
    ValueError: If `opt` is a supported string but `learning_rate` was not
      specified.
    ValueError: If `opt` is none of the above types.
  """
    if isinstance(opt, six.string_types):
        if opt in six.iterkeys(_OPTIMIZER_CLS_NAMES):
            if not learning_rate:
                raise ValueError('learning_rate must be specified when opt is string.')
            return _OPTIMIZER_CLS_NAMES[opt](learning_rate=learning_rate)
        raise ValueError('Unsupported optimizer name: {}. Supported names are: {}'.format(opt, tuple(sorted(six.iterkeys(_OPTIMIZER_CLS_NAMES)))))
    if callable(opt):
        opt = opt()
    if not isinstance(opt, tf.compat.v1.train.Optimizer):
        raise ValueError('The given object is not an Optimizer instance. Given: {}'.format(opt))
    return opt