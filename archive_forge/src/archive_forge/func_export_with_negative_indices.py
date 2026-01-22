import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_with_negative_indices() -> None:
    axisValue = 1
    on_value = 3
    off_value = 1
    output_type = np.float32
    node = onnx.helper.make_node('OneHot', inputs=['indices', 'depth', 'values'], outputs=['y'], axis=axisValue)
    indices = np.array([0, -7, -8], dtype=np.int64)
    depth = np.float32(10)
    values = np.array([off_value, on_value], dtype=output_type)
    y = one_hot(indices, depth, axis=axisValue, dtype=output_type)
    y = y * (on_value - off_value) + off_value
    expect(node, inputs=[indices, depth, values], outputs=[y], name='test_onehot_negative_indices')