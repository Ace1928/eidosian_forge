import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_pool_common import (
@staticmethod
def export_averagepool_3d_dilations_large() -> None:
    x_shape = (32, 32, 32)
    dilations = (2, 2, 2)
    kernel_shape = (5, 5, 5)
    strides = (3, 3, 3)
    count_include_pad = 0
    for count_include_pad in (0, 1):
        for ceil_mode in (True, False):
            node = onnx.helper.make_node('AveragePool', inputs=['x'], outputs=['y'], kernel_shape=kernel_shape, strides=strides, dilations=dilations, count_include_pad=count_include_pad, ceil_mode=ceil_mode)
            x = np.random.randn(1, 1, *x_shape).astype(np.float32)
            out_shape, pads = get_output_shape_explicit_padding(None, x_shape, kernel_shape, strides, dilations=dilations, ceil_mode=ceil_mode)
            padded = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[3]), (pads[1], pads[4]), (pads[2], pads[5])), mode='constant', constant_values=0 if count_include_pad == 1 else np.nan)
            y = pool(padded, (1, 1, *x_shape), kernel_shape, strides, out_shape, 'AVG', pads=pads, dilations=dilations, count_include_pad=count_include_pad)
            test_name = f'test_averagepool_3d_dilations_large_count_include_pad_is_{count_include_pad}_ceil_mode_is_{ceil_mode}'
            expect(node, inputs=[x], outputs=[y], name=test_name)