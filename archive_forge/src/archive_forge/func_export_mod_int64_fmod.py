import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_mod_int64_fmod() -> None:
    node = onnx.helper.make_node('Mod', inputs=['x', 'y'], outputs=['z'], fmod=1)
    x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int64)
    y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int64)
    z = np.fmod(x, y)
    expect(node, inputs=[x, y], outputs=[z], name='test_mod_int64_fmod')