from typing import Tuple
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_einsum_batch_matmul() -> None:
    Eqn = 'bij, bjk -> bik'
    node = onnx.helper.make_node('Einsum', inputs=['x', 'y'], outputs=['z'], equation=Eqn)
    X = np.random.randn(5, 2, 3)
    Y = np.random.randn(5, 3, 4)
    Z = einsum_reference_implementation(Eqn, (X, Y))
    expect(node, inputs=[X, Y], outputs=[Z], name='test_einsum_batch_matmul')