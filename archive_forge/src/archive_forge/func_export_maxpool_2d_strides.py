import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_pool_common import (
@staticmethod
def export_maxpool_2d_strides() -> None:
    """input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 10, 10]
        """
    node = onnx.helper.make_node('MaxPool', inputs=['x'], outputs=['y'], kernel_shape=[5, 5], strides=[3, 3])
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    pads = None
    kernel_shape = (5, 5)
    strides = (3, 3)
    out_shape, pads = get_output_shape_explicit_padding(pads, x_shape[2:], kernel_shape, strides)
    padded = x
    y = pool(padded, x_shape, kernel_shape, strides, out_shape, 'MAX')
    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_strides')