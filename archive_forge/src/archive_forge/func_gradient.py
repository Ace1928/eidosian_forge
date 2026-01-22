import collections
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
def gradient():
    with backprop.GradientTape() as tape:
        tape.watch(in_op)
        op_output = nn_ops.conv2d_transpose_v2(in_op, filter_op, out_shape, strides=1, padding='SAME', data_format='NHWC', dilations=[1, rate, rate, 1])
        gradient_injector_output = op_output * upstream_gradients
    return tape.gradient(gradient_injector_output, [in_op])[0]