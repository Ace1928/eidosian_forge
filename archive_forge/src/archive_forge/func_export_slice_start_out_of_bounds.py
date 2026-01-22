import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_slice_start_out_of_bounds() -> None:
    node = onnx.helper.make_node('Slice', inputs=['x', 'starts', 'ends', 'axes', 'steps'], outputs=['y'])
    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([1000], dtype=np.int64)
    ends = np.array([1000], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    y = x[:, 1000:1000]
    expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y], name='test_slice_start_out_of_bounds')