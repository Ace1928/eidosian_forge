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
@unittest.skipIf(version_utils.numpy_older_than('1.21.5'), 'op_dft and op_stft requires numpy >= 1.21.5')
def test_stft(self):
    signal = make_tensor_value_info('signal', TensorProto.FLOAT, [None, None, None])
    frame_step = make_tensor_value_info('frame_step', TensorProto.INT64, [None])
    frame_length = make_tensor_value_info('frame_length', TensorProto.INT64, [None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
    node = make_node('STFT', ['signal', 'frame_step', '', 'frame_length'], ['Y'])
    graph = make_graph([node], 'g', [signal, frame_step, frame_length], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid('', 17)])
    feeds = {'signal': np.arange(128).reshape((1, 128, 1)).astype(np.float32), 'frame_step': np.array(8, dtype=np.int64), 'frame_length': np.array(16, dtype=np.int64)}
    signal = feeds['signal']
    frame_length = int(feeds['frame_length'])
    frame_step = int(feeds['frame_step'])
    onesided_length = frame_length // 2 + 1
    nstfts = (feeds['signal'].shape[1] - frame_length) // frame_step + 1
    expected = np.empty([1, nstfts, onesided_length, 2], dtype=np.float32)
    for i in range(nstfts):
        start = i * frame_step
        stop = i * frame_step + frame_length
        complex_out = np.fft.fft(signal[0, start:stop, 0])
        c_out = complex_out[0:onesided_length]
        expected[0, i] = np.stack((c_out.real, c_out.imag), axis=1)
    ref1 = ReferenceEvaluator(onnx_model)
    got1 = ref1.run(None, feeds)
    assert_allclose(expected, got1[0])