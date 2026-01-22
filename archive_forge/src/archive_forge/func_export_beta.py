from typing import Optional
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_beta() -> None:
    node = onnx.helper.make_node('Gemm', inputs=['a', 'b', 'c'], outputs=['y'], beta=0.5)
    a = np.random.ranf([2, 7]).astype(np.float32)
    b = np.random.ranf([7, 4]).astype(np.float32)
    c = np.random.ranf([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, beta=0.5)
    expect(node, inputs=[a, b, c], outputs=[y], name='test_gemm_beta')