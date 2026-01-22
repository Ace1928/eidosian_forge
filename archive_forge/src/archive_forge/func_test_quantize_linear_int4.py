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
@parameterized.parameterized.expand([(TensorProto.UINT4, [-1, 0, 1.5, 2, 3.3, 10, 20, 40], [0, 0, 2, 2, 4, 10, 20, 30]), (TensorProto.UINT4, [-1, 0, 1.5, 2, 3.3, 10, 40], [0, 0, 2, 2, 4, 10, 30]), (TensorProto.UINT4, [0], [0]), (TensorProto.INT4, [-20, -14.5, 0, 1.5, 2, 3.3, 10, 20], [-16, -14, 0, 2, 2, 4, 10, 14]), (TensorProto.INT4, [-20, -14.5, 0, 1.5, 2, 3.3, 10], [-16, -14, 0, 2, 2, 4, 10]), (TensorProto.INT4, [0], [0])])
@unittest.skipIf(version_utils.numpy_older_than('1.22.0'), 'The test requires numpy 1.22.0 or later')
def test_quantize_linear_int4(self, qtype, data, expected):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
    model = make_model(make_graph([make_node('Constant', [], ['scale'], value=make_tensor('scale', TensorProto.FLOAT, [1], [2.0])), make_node('Constant', [], ['zero'], value=make_tensor('zero', qtype, [1], [0])), make_node('QuantizeLinear', ['X', 'scale', 'zero'], ['T']), make_node('DequantizeLinear', ['T', 'scale'], ['Y'], axis=0)], 'g', [X], [Y]))
    ref = ReferenceEvaluator(model)
    got = ref.run(None, {'X': np.asarray(data)})
    assert_allclose(expected, got[0])