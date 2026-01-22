import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_opset19() -> None:
    node = onnx.helper.make_node('DFT', inputs=['x'], outputs=['y'], axis=1)
    x = np.arange(0, 100).reshape(10, 10).astype(np.float32)
    y = np.fft.fft(x, axis=0)
    x = x.reshape(1, 10, 10, 1)
    y = np.stack((y.real, y.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
    expect(node, inputs=[x], outputs=[y], name='test_dft_opset19', opset_imports=[onnx.helper.make_opsetid('', 19)])
    node = onnx.helper.make_node('DFT', inputs=['x'], outputs=['y'], axis=2)
    x = np.arange(0, 100).reshape(10, 10).astype(np.float32)
    y = np.fft.fft(x, axis=1)
    x = x.reshape(1, 10, 10, 1)
    y = np.stack((y.real, y.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
    expect(node, inputs=[x], outputs=[y], name='test_dft_axis_opset19', opset_imports=[onnx.helper.make_opsetid('', 19)])
    node = onnx.helper.make_node('DFT', inputs=['x'], outputs=['y'], inverse=1, axis=1)
    x = np.arange(0, 100, dtype=np.complex64).reshape(10, 10)
    y = np.fft.ifft(x, axis=0)
    x = np.stack((x.real, x.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
    y = np.stack((y.real, y.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
    expect(node, inputs=[x], outputs=[y], name='test_dft_inverse_opset19', opset_imports=[onnx.helper.make_opsetid('', 19)])