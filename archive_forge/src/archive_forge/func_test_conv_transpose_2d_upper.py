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
def test_conv_transpose_2d_upper(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None, None])
    W = make_tensor_value_info('W', TensorProto.FLOAT, [None, None, None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
    node = make_node('ConvTranspose', ['X', 'W', 'B'], ['Y'], auto_pad='SAME_UPPER', strides=[2, 2])
    graph = make_graph([node], 'g', [X, W, B], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
    feeds = {'X': np.arange(1 * 1 * 3 * 3).reshape((1, 1, 3, 3)).astype(np.float32), 'W': np.arange(1 * 2 * 3 * 3).reshape((1, 2, 3, 3)).astype(np.float32), 'B': np.array([0, 0, 0, 0], dtype=np.float32)}
    expected = np.array([[[[0, 0, 0, 1, 2, 2], [0, 0, 3, 4, 11, 8], [0, 3, 12, 11, 28, 19], [9, 12, 27, 16, 35, 20], [18, 27, 60, 35, 76, 43], [18, 24, 51, 28, 59, 32]], [[0, 0, 9, 10, 29, 20], [0, 0, 12, 13, 38, 26], [27, 30, 84, 56, 136, 82], [36, 39, 90, 52, 116, 65], [99, 108, 240, 134, 292, 160], [72, 78, 168, 91, 194, 104]]]], dtype=np.float32)
    ref1 = ReferenceEvaluator(onnx_model)
    got1 = ref1.run(None, feeds)
    assert_allclose(expected, got1[0])