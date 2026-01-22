import itertools
import math
import sys
import unittest
from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
from os import getenv
from textwrap import dedent
from typing import Sequence, Tuple
import numpy as np
import parameterized
import version_utils
from numpy.testing import assert_allclose
import onnx.reference.custom_element_types as custom
from onnx import (
from onnx.backend.test.case.node.roialign import get_roi_align_input_values
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32, from_array
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun, OpRunExpand
from onnx.reference.ops import load_op
from onnx.reference.ops._op_common_indices import _get_indices, _is_out
from onnx.reference.ops._op_list import Cast_19, Celu
from onnx.reference.ops.aionnx_preview_training._op_list import Adam
from onnx.reference.ops.op_celu import _vcelu1
from onnx.reference.ops.op_col2im import (
from onnx.reference.ops.op_conv import Conv, _conv_implementation
from onnx.reference.ops_optimized import Conv as ConvOptimized
from onnx.reference.ops_optimized.op_conv_optimized import _conv_implementation_im2col
@parameterized.parameterized.expand([(4 * np.arange(12).reshape(3, 4), np.arange(1, 7).reshape(3, 2), np.zeros((3, 2)), 1, 2, [[0, 4, 4, 6], [5, 7, 6, 7], [6, 7, 7, 7]]), (4 * np.arange(12).reshape(3, 4), np.arange(1, 7).reshape(3, 2), np.ones((3, 2)), 1, 2, [[1, 5, 5, 7], [6, 8, 7, 8], [7, 8, 8, 8]]), (np.arange(24).reshape(3, 8), [[0.25, 0.5, 1], [0.25, 0.5, 1], [0.25, 0.5, 1]], np.zeros((3, 3)), 1, 3, [[0, 4, 8, 6, 8, 10, 6, 7], [32, 36, 40, 22, 24, 26, 14, 15], [64, 68, 72, 38, 40, 42, 22, 23]]), (np.arange(6), [0.25, 0.5], [-1, -2], 0, 3, [-1, 3, 7, 4, 6, 8]), (np.ones((9, 12)), np.ones((3, 4)), np.zeros((3, 4)), 0, 3, None), (np.ones((3, 4, 5, 6)), np.ones((3, 4)), np.zeros((3, 4)), 2, 2, None)])
def test_blocked_quantize_linear(self, x, scale, zero_point, axis, block_size, expected):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
    Y = make_tensor_value_info('Y', TensorProto.INT8, [None])
    scale_data = np.array(scale, dtype=np.float32)
    zp_data = np.array(zero_point, dtype=np.int8)
    model = make_model(make_graph([make_node('QuantizeLinear', ['X', 'scale', 'zero'], ['Y'], axis=axis, block_size=block_size)], 'g', [X], [Y], [make_tensor('scale', TensorProto.FLOAT, scale_data.shape, scale_data), make_tensor('zero', TensorProto.INT8, scale_data.shape, zp_data)]))
    ref = ReferenceEvaluator(model)
    data = np.array(x, dtype=np.float32)
    if expected is not None:
        expected = np.array(expected, dtype=np.int8)
        got = ref.run(None, {'X': data})
        assert_allclose(expected, got[0])
    else:
        with self.assertRaises(ValueError):
            ref.run(None, {'X': data})