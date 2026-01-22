import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_equal_broadcast() -> None:
    node = onnx.helper.make_node('Equal', inputs=['x', 'y'], outputs=['z'])
    x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
    y = (np.random.randn(5) * 10).astype(np.int32)
    z = np.equal(x, y)
    expect(node, inputs=[x, y], outputs=[z], name='test_equal_bcast')