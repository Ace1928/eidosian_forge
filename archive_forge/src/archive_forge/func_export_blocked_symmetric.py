import numpy as np
import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import make_tensor
@staticmethod
def export_blocked_symmetric() -> None:
    node = onnx.helper.make_node('QuantizeLinear', inputs=['x', 'y_scale'], outputs=['y'], axis=1, block_size=2, output_dtype=TensorProto.INT16)
    x = np.array([[6.0, -8, -10, 5.0], [1.0, 8.0, 4.0, 5.0], [0.0, 20.0, 10.0, 4.0]], dtype=np.float32)
    y_scale = np.array([[1.5, 2.5], [3.0, 4.9], [5.1, 6.9]], dtype=np.float32)
    block_axis = 1
    assert all((x.shape[i] == y_scale.shape[i] for i in range(len(x.shape)) if i != block_axis))
    assert x.shape[block_axis] % y_scale.shape[block_axis] == 0
    repeats = x.shape[block_axis] // y_scale.shape[block_axis]
    y_scale_elementwise = np.repeat(y_scale, repeats=repeats, axis=block_axis)
    y_val = np.clip(np.rint(x / y_scale_elementwise), a_min=-32768, a_max=32767).astype(np.int16)
    y = make_tensor('y', TensorProto.INT16, x.shape, y_val)
    expect(node, inputs=[x, y_scale], outputs=[y], name='test_quantizelinear_blocked_symmetric')