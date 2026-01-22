import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_constant_pad_axes() -> None:
    node = onnx.helper.make_node('Pad', inputs=['x', 'pads', 'value', 'axes'], outputs=['y'], mode='constant')
    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    pads = np.array([0, 3, 0, 4]).astype(np.int64)
    value = np.float32(1.2)
    axes = np.array([1, 3], dtype=np.int64)
    y = pad_impl(x, pads, 'constant', 1.2, [1, 3])
    expect(node, inputs=[x, pads, value, axes], outputs=[y], name='test_constant_pad_axes')