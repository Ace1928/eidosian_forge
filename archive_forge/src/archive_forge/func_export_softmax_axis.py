import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_softmax_axis() -> None:
    x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
    y = softmax(x)
    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'])
    expect(node, inputs=[x], outputs=[y], name='test_softmax_large_number')
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'], axis=0)
    y = softmax(x, axis=0)
    expect(node, inputs=[x], outputs=[y], name='test_softmax_axis_0')
    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'], axis=1)
    y = softmax(x, axis=1)
    expect(node, inputs=[x], outputs=[y], name='test_softmax_axis_1')
    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'], axis=2)
    y = softmax(x, axis=2)
    expect(node, inputs=[x], outputs=[y], name='test_softmax_axis_2')
    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'], axis=-1)
    y = softmax(x, axis=-1)
    expect(node, inputs=[x], outputs=[y], name='test_softmax_negative_axis')
    node = onnx.helper.make_node('Softmax', inputs=['x'], outputs=['y'])
    expect(node, inputs=[x], outputs=[y], name='test_softmax_default_axis')