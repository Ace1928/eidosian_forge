import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_nd() -> None:
    node = onnx.helper.make_node('Det', inputs=['x'], outputs=['y'])
    x = np.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]).astype(np.float32)
    y = np.linalg.det(x)
    expect(node, inputs=[x], outputs=[y], name='test_det_nd')