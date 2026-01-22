import collections
import functools
import re
import string
import numpy as np
import opt_einsum
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_special_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('math.lbeta', v1=['math.lbeta', 'lbeta'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('lbeta')
def lbeta(x, name=None):
    """Computes \\\\(ln(|Beta(x)|)\\\\), reducing along the last dimension.

  Given one-dimensional $z = [z_1,...,z_K]$, we define

  $$Beta(z) = \\frac{\\prod_j \\Gamma(z_j)}{\\Gamma(\\sum_j z_j)},$$

  where $\\Gamma$ is the gamma function.

  And for $n + 1$ dimensional $x$ with shape $[N_1, ..., N_n, K]$, we define

  $$lbeta(x)[i_1, ..., i_n] = \\log{|Beta(x[i_1, ..., i_n, :])|}.$$

  In other words, the last dimension is treated as the $z$ vector.

  Note that if $z = [u, v]$, then

  $$Beta(z) = \\frac{\\Gamma(u)\\Gamma(v)}{\\Gamma(u + v)}
    = \\int_0^1 t^{u-1} (1 - t)^{v-1} \\mathrm{d}t,$$

  which defines the traditional bivariate beta function.

  If the last dimension is empty, we follow the convention that the sum over
  the empty set is zero, and the product is one.

  Args:
    x: A rank `n + 1` `Tensor`, `n >= 0` with type `float`, or `double`.
    name: A name for the operation (optional).

  Returns:
    The logarithm of \\\\(|Beta(x)|\\\\) reducing along the last dimension.
  """
    with ops.name_scope(name, 'lbeta', [x]):
        x = ops.convert_to_tensor(x, name='x')
        log_prod_gamma_x = math_ops.reduce_sum(math_ops.lgamma(x), axis=[-1])
        sum_x = math_ops.reduce_sum(x, axis=[-1])
        log_gamma_sum_x = math_ops.lgamma(sum_x)
        result = log_prod_gamma_x - log_gamma_sum_x
        return result