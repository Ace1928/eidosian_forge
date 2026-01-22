from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
from absl import logging
import six
import tensorflow as tf
def get_optimizer_instance_v2(opt, learning_rate=None):
    """Returns an optimizer_v2.OptimizerV2 instance.

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
  """
    if isinstance(opt, six.string_types):
        if opt in six.iterkeys(_OPTIMIZER_CLS_NAMES_V2):
            if not learning_rate:
                if _optimizer_has_default_learning_rate(_OPTIMIZER_CLS_NAMES_V2[opt]):
                    return _OPTIMIZER_CLS_NAMES_V2[opt]()
                else:
                    return _OPTIMIZER_CLS_NAMES_V2[opt](learning_rate=_LEARNING_RATE)
            return _OPTIMIZER_CLS_NAMES_V2[opt](learning_rate=learning_rate)
        raise ValueError('Unsupported optimizer name: {}. Supported names are: {}'.format(opt, tuple(sorted(six.iterkeys(_OPTIMIZER_CLS_NAMES_V2)))))
    if callable(opt):
        opt = opt()
    if isinstance(opt, tf.keras.optimizers.experimental.Optimizer):
        if tf.executing_eagerly():
            logging.warning('You are using `tf.keras.optimizers.experimental.Optimizer` in TF estimator, which only supports `tf.keras.optimizers.legacy.Optimizer`. Automatically converting your optimizer to `tf.keras.optimizers.legacy.Optimizer`.')
            opt = tf.keras.__internal__.optimizers.convert_to_legacy_optimizer(opt)
        else:
            raise ValueError(f'Please set your optimizer as an instance of `tf.keras.optimizers.legacy.Optimizer`, e.g., `tf.keras.optimizers.legacy.{opt.__class__.__name__}`.Received optimizer type: {type(opt)}.')
    if not isinstance(opt, (tf.keras.optimizers.legacy.Optimizer, tf.keras.optimizers.Optimizer)):
        raise ValueError('The given object is not a tf.keras.optimizers.Optimizer instance. Given: {}'.format(opt))
    return opt