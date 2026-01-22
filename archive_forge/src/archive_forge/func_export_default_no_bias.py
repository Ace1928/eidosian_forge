from typing import Optional
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_default_no_bias() -> None:
    node = onnx.helper.make_node('Gemm', inputs=['a', 'b'], outputs=['y'])
    a = np.random.ranf([2, 10]).astype(np.float32)
    b = np.random.ranf([10, 3]).astype(np.float32)
    y = gemm_reference_implementation(a, b)
    expect(node, inputs=[a, b], outputs=[y], name='test_gemm_default_no_bias')