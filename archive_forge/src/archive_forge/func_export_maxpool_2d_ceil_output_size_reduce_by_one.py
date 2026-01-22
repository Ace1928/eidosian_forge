import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_pool_common import (
@staticmethod
def export_maxpool_2d_ceil_output_size_reduce_by_one() -> None:
    """input_shape: [1, 1, 2, 2]
        output_shape: [1, 1, 1, 1]
        """
    node = onnx.helper.make_node('MaxPool', inputs=['x'], outputs=['y'], kernel_shape=[1, 1], strides=[2, 2], ceil_mode=True)
    x = np.array([[[[1, 2], [3, 4]]]]).astype(np.float32)
    y = np.array([[[[1]]]]).astype(np.float32)
    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_ceil_output_size_reduce_by_one')