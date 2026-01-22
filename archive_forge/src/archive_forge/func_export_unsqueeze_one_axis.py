import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_unsqueeze_one_axis() -> None:
    x = np.random.randn(3, 4, 5).astype(np.float32)
    for i in range(x.ndim):
        axes = np.array([i]).astype(np.int64)
        node = onnx.helper.make_node('Unsqueeze', inputs=['x', 'axes'], outputs=['y'])
        y = np.expand_dims(x, axis=i)
        expect(node, inputs=[x, axes], outputs=[y], name='test_unsqueeze_axis_' + str(i))