import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_pool_common import (
@staticmethod
def export_lppool_2d_dilations() -> None:
    """input_shape: [1, 1, 4, 4]
        output_shape: [1, 1, 2, 2]
        """
    p = 2
    node = onnx.helper.make_node('LpPool', inputs=['x'], outputs=['y'], kernel_shape=[2, 2], strides=[1, 1], dilations=[2, 2], p=p)
    x = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]).astype(np.float32)
    y = np.array([[[[14.560219778561036, 16.24807680927192], [21.633307652783937, 23.49468024894146]]]]).astype(np.float32)
    expect(node, inputs=[x], outputs=[y], name='test_lppool_2d_dilations')