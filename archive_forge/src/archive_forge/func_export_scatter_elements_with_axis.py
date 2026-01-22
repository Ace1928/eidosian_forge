import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_scatter_elements_with_axis() -> None:
    axis = 1
    node = onnx.helper.make_node('ScatterElements', inputs=['data', 'indices', 'updates'], outputs=['y'], axis=axis)
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    indices = np.array([[1, 3]], dtype=np.int64)
    updates = np.array([[1.1, 2.1]], dtype=np.float32)
    y = scatter_elements(data, indices, updates, axis)
    expect(node, inputs=[data, indices, updates], outputs=[y], name='test_scatter_elements_with_axis')