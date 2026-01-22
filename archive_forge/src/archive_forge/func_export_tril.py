import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_tril() -> None:
    node = onnx.helper.make_node('Trilu', inputs=['x'], outputs=['y'], upper=0)
    x = np.random.randint(10, size=(4, 5)).astype(np.int64)
    y = tril_reference_implementation(x)
    expect(node, inputs=[x], outputs=[y], name='test_tril')