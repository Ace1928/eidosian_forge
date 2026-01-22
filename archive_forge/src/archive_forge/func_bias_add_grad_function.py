import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
def bias_add_grad_function(upstream_gradients):
    with backprop.GradientTape() as tape:
        tape.watch(bias_tensor)
        bias_add_output = bias_add(input_tensor, bias_tensor)
        gradient_injector_output = bias_add_output * upstream_gradients
        return tape.gradient(gradient_injector_output, bias_tensor)