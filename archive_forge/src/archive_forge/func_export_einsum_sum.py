from typing import Tuple
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_einsum_sum() -> None:
    Eqn = 'ij->i'
    node = onnx.helper.make_node('Einsum', inputs=['x'], outputs=['y'], equation=Eqn)
    X = np.random.randn(3, 4)
    Z = einsum_reference_implementation(Eqn, (X,))
    expect(node, inputs=[X], outputs=[Z], name='test_einsum_sum')