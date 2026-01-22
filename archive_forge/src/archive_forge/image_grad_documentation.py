from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
The gradients for `rgb_to_hsv` operation.

  This function is a piecewise continuous function as defined here:
  https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB
  We perform the multivariate derivative and compute all partial derivatives
  separately before adding them in the end. Formulas are given before each
  partial derivative calculation.

  Args:
    op: The `rgb_to_hsv` `Operation` that we are differentiating.
    grad: Gradient with respect to the output of the `rgb_to_hsv` op.

  Returns:
    Gradients with respect to the input of `rgb_to_hsv`.
  