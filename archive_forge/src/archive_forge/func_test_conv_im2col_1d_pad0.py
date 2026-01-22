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
def test_conv_im2col_1d_pad0(self):
    feeds = {'X': np.arange(2 * 4 * 3).reshape((2, 4, -1)).astype(np.float32) + 1, 'W': np.arange(2 * 4 * 3).reshape((-1, 4, 3)).astype(np.float32), 'B': np.zeros((1,), dtype=np.float32)}
    kwargs = dict(group=1, dilations=[1], kernel_shape=[3], pads=[0, 0], strides=[1], auto_pad='NOTSET')
    expected = _conv_implementation(**feeds, **kwargs)
    got = _conv_implementation_im2col(**feeds, **kwargs)
    assert_allclose(expected, got)