import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_slice_negative_axes() -> None:
    node = onnx.helper.make_node('Slice', inputs=['x', 'starts', 'ends', 'axes'], outputs=['y'])
    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    axes = np.array([0, -2, -1], dtype=np.int64)
    y = x[:, :, 3:4]
    expect(node, inputs=[x, starts, ends, axes], outputs=[y], name='test_slice_negative_axes')