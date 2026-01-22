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
def test_float8_4_types(self):
    x = np.array([0.4068359375, 352, 416, 336, 304, 272, -248, -100, 0.0001, 0.01, 416, 432, 100000.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    expected = {TensorProto.FLOAT8E4M3FN: np.array([0.40625, 352.0, 416.0, 320.0, 320.0, 256.0, -256.0, -96.0, 0.0, 0.009765625, 416.0, 448.0, 448.0, 448.0, -448.0, np.nan], dtype=np.float32), TensorProto.FLOAT8E4M3FNUZ: np.array([0.40625, 240.0, 240.0, 240.0, 240.0, 240.0, -240.0, -96.0, 0.0, 0.009765625, 240.0, 240.0, 240.0, 240.0, -240.0, np.nan], dtype=np.float32), TensorProto.FLOAT8E5M2: np.array([0.4375, 384.0, 384.0, 320.0, 320.0, 256.0, -256.0, -96.0, 0.0001068115234375, 0.009765625, 384.0, 448.0, 57344.0, 57344.0, -57344.0, np.nan], dtype=np.float32), TensorProto.FLOAT8E5M2FNUZ: np.array([0.4375, 384.0, 384.0, 320.0, 320.0, 256.0, -256.0, -96.0, 0.00010681152, 0.009765625, 384.0, 448.0, 57344.0, 57344.0, -57344.0, np.nan], dtype=np.float32)}

    def model_cast_cast(to):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('Cast', ['X'], ['T'], to=to)
        node2 = make_node('Cast', ['T'], ['Y'], to=TensorProto.FLOAT)
        graph = make_graph([node1, node2], 'lr', [X], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        return onnx_model
    for to, expect in expected.items():
        with self.subTest(to=to):
            onnx_model = model_cast_cast(to)
            ref = ReferenceEvaluator(onnx_model)
            y = ref.run(None, {'X': x})[0]
            assert_allclose(expect, y)
            self.assertEqual(expect.shape, y.shape)
            self.assertEqual(expect.dtype, y.dtype)