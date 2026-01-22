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
@parameterized.parameterized.expand(itertools.product([('ReduceMin', [np.array([[np.nan, np.nan], [14.422706, 18.80527]], dtype=np.float32), np.array([[2, 15], [10, 4]], dtype=np.int64)]), ('ReduceL1', [np.array([[2.2367053, 2.3516612], [4.076292, 4.2970634]], dtype=np.float32), np.array([[18, 6], [13, 6]], dtype=np.int64)]), ('ReduceL2', [np.array([[1.80155, 1.8169948], [2.9928076, 3.1205883]], dtype=np.float32), np.array([[11, 18], [13, 6]], dtype=np.int64)]), ('ReduceLogSum', [np.array([[0.9497848, 1.1872643], [1.6764175, 1.70759]], dtype=np.float32), np.array([[6, 18], [13, 6]], dtype=np.int64)]), ('ReduceLogSumExp', [np.array([[1.6005973, 1.7445935], [2.5616229, 2.6539795]], dtype=np.float32), np.array([[13, 6], [13, 6]], dtype=np.int64)]), ('ReduceMax', [np.array([[1.4217108, 1.5069536], [2.453826, 2.5041783]], dtype=np.float32), np.array([[13, 11], [13, 11]], dtype=np.int64)]), ('ReduceMean', [np.array([[0.39247903, 0.78497636], [2.038146, 2.1485317]], dtype=np.float32), np.array([[13, 6], [13, 6]], dtype=np.int64)]), ('ReduceSumSquare', [np.array([[3.2455828, 3.3014696], [8.956896, 9.7380705]], dtype=np.float32), np.array([[11, 18], [13, 6]], dtype=np.int64)]), ('ReduceProd', [np.array([[np.nan, np.nan], [14.422706, 18.80527]], dtype=np.float32), np.array([[2, 15], [13, 6]], dtype=np.int64)])], [17, 18]))
def test_op_reduce(self, reduce_op_expected, opset: int):
    reduce_op, expected = reduce_op_expected
    X = np.arange(8).reshape((-1, 4)).astype(np.float32)
    results = {}
    model = self._cdist_model(opset, reduce_op)
    sess = ReferenceEvaluator(model)
    got = sess.run(None, {'input': X})
    results['ref', opset] = got
    cl = [n for n in sess.rt_nodes_[0].body.rt_nodes_ if n.__class__.__name__.startswith(reduce_op)]
    schema = cl[0]._schema
    new_cl = type(reduce_op, (cl[0].__class__,), {'op_schema': schema})
    sess = ReferenceEvaluator(model, new_ops=[new_cl])
    got = sess.run(None, {'input': X})
    results['ref_cl', opset] = got
    baseline = 'constant'
    for k, v in results.items():
        for a, b in zip(reversed(expected), reversed(v)):
            if a.shape != b.shape:
                raise AssertionError(f'Shape mismatch for {reduce_op!r}, {baseline}:{a.shape} != {k}:{b.shape}.')
            diff = np.abs(a - b).max()
            if diff > 1e-06:
                raise AssertionError(f'Discrepancies (max={diff}) for {reduce_op!r}, {baseline} != {k}\n{a}\n!=\n{b}')