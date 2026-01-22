import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_long() -> None:
    node = onnx.helper.make_node('Where', inputs=['condition', 'x', 'y'], outputs=['z'])
    condition = np.array([[1, 0], [1, 1]], dtype=bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[9, 8], [7, 6]], dtype=np.int64)
    z = np.where(condition, x, y)
    expect(node, inputs=[condition, x, y], outputs=[z], name='test_where_long_example')