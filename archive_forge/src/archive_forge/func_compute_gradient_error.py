import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['test.compute_gradient_error'])
@deprecation.deprecated(date=None, instructions='Use tf.test.compute_gradient in 2.0, which has better support for functions. Note that the two versions have different usage, so code change is needed.')
def compute_gradient_error(x, x_shape, y, y_shape, x_init_value=None, delta=0.001, init_targets=None, extra_feed_dict=None):
    """Computes the gradient error.

  Computes the maximum error for dy/dx between the computed Jacobian and the
  numerically estimated Jacobian.

  This function will modify the tensors passed in as it adds more operations
  and hence changing the consumers of the operations of the input tensors.

  This function adds operations to the current session. To compute the error
  using a particular device, such as a GPU, use the standard methods for
  setting a device (e.g. using with sess.graph.device() or setting a device
  function in the session constructor).

  Args:
    x: a tensor or list of tensors
    x_shape: the dimensions of x as a tuple or an array of ints. If x is a list,
    then this is the list of shapes.
    y: a tensor
    y_shape: the dimensions of y as a tuple or an array of ints.
    x_init_value: (optional) a numpy array of the same shape as "x"
      representing the initial value of x. If x is a list, this should be a list
      of numpy arrays.  If this is none, the function will pick a random tensor
      as the initial value.
    delta: (optional) the amount of perturbation.
    init_targets: list of targets to run to initialize model params.
    extra_feed_dict: dict that allows fixing specified tensor values
      during the Jacobian calculation.

  Returns:
    The maximum error in between the two Jacobians.
  """
    grad = compute_gradient(x, x_shape, y, y_shape, x_init_value, delta, init_targets, extra_feed_dict=extra_feed_dict)
    return _compute_error(grad)