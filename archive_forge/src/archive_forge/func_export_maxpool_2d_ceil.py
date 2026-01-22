import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_pool_common import (
@staticmethod
def export_maxpool_2d_ceil() -> None:
    """input_shape: [1, 1, 4, 4]
        output_shape: [1, 1, 2, 2]
        """
    node = onnx.helper.make_node('MaxPool', inputs=['x'], outputs=['y'], kernel_shape=[3, 3], strides=[2, 2], ceil_mode=True)
    x = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]).astype(np.float32)
    y = np.array([[[[11, 12], [15, 16]]]]).astype(np.float32)
    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_ceil')