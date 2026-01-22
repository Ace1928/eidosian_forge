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
@skip_if_no_torch
def test_col2im(self):
    import torch
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None])
    IS = make_tensor_value_info('I', TensorProto.INT64, [None])
    BS = make_tensor_value_info('B', TensorProto.INT64, [None])
    node = make_node('Col2Im', ['X', 'I', 'B'], ['Y'], pads=[0, 0, 0, 0], strides=[1, 1], dilations=[1, 1])
    graph = make_graph([node], 'g', [X, IS, BS], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
    sess = ReferenceEvaluator(onnx_model)
    X = np.array([[[1.0, 6.0, 11.0, 16.0, 21.0], [2.0, 7.0, 12.0, 17.0, 22.0], [3.0, 8.0, 13.0, 18.0, 23.0], [4.0, 9.0, 14.0, 19.0, 24.0], [5.0, 0.0, 15.0, 20.0, 25.0]]]).astype(np.float32)
    image_shape = np.array([5, 5]).astype(np.int64)
    block_shape = np.array([1, 5]).astype(np.int64)
    fold = torch.nn.Fold(output_size=tuple(image_shape), kernel_size=block_shape)
    got = sess.run(None, {'X': X, 'B': block_shape, 'I': image_shape})
    output = fold(torch.from_numpy(X)).numpy()
    assert_allclose(output, got[0])