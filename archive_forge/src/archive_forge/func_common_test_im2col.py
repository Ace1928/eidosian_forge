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
def common_test_im2col(self, kernel_shape, pads, strides, dilations):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None, None])
    Y1 = make_tensor_value_info('Y1', TensorProto.FLOAT, [None, None, None, None])
    Y2 = make_tensor_value_info('Y2', TensorProto.FLOAT, [None, None, None, None])
    W = make_tensor_value_info('W', TensorProto.FLOAT, [None, None, None, None])
    node = make_node('Conv', ['X', 'W'], ['Y1'], pads=pads, strides=strides, dilations=dilations)
    node_shape = make_node('Shape', ['W'], ['shape'])
    node_im = make_node('Im2Col', ['X', 'shape'], ['xim'], pads=pads, strides=strides, dilations=dilations, domain='experimental')
    node_flat = make_node('Flatten', ['W'], ['wflat'])
    node_gem = make_node('MatMul', ['wflat', 'xim'], ['Y2'])
    graph = make_graph([node, node_shape, node_im, node_flat, node_gem], 'g', [X, W], [Y1, Y2])
    onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16), make_opsetid('experimental', 1)])
    graph_conv = make_graph([node], 'g', [X, W], [Y1])
    onnx_model_conv = make_model_gen_version(graph_conv, opset_imports=[make_opsetid('', 16)])
    sess = ReferenceEvaluator(onnx_model)
    try:
        sess_conv = run_ort_inference(onnx_model_conv)
        if sess_conv is None:
            return
    except ImportError:
        sess_conv = None
    sH, sW = (7, 7)
    nker = np.prod(kernel_shape)
    for i in range(sH):
        for j in range(sW):
            X = np.zeros((1, 1, sH, sW), dtype=np.float32)
            X[0, 0, i, j] = 1.0
            W = np.zeros((1, 1, *kernel_shape), dtype=np.float32)
            W[0, 0, :, :] = np.minimum(2 ** np.arange(nker).reshape((kernel_shape[0], -1)), 256)
            got = sess.run(None, {'X': X, 'W': W})
            if sess_conv is not None:
                ort_res = sess_conv.run(None, {'X': X, 'W': W})[0]
                assert_allclose(got[1].ravel(), ort_res.ravel())
            try:
                assert_allclose(got[0].ravel(), got[1].ravel())
            except AssertionError as e:
                raise AssertionError(f'Discrepancies: pads={pads}, dilations={dilations}, strides={strides}, kernel_shape={kernel_shape}\n{got[0]}\n!=\n{got[1]}') from e