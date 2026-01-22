import numpy as np
import onnx
from onnx import helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_default_mask_ratio() -> None:
    seed = np.int64(0)
    node = onnx.helper.make_node('Dropout', inputs=['x', 'r'], outputs=['y', 'z'], seed=seed)
    r = np.float32(0.1)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y, z = dropout(x, r, return_mask=True)
    expect(node, inputs=[x, r], outputs=[y, z], name='test_dropout_default_mask_ratio')