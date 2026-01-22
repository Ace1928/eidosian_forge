import numpy as np
import onnx
from onnx import helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_random_old() -> None:
    node = onnx.helper.make_node('Dropout', inputs=['x'], outputs=['y'], ratio=0.2)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = x
    expect(node, inputs=[x], outputs=[y], name='test_dropout_random_old', opset_imports=[helper.make_opsetid('', 11)])