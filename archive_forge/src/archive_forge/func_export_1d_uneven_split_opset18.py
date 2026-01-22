import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_1d_uneven_split_opset18() -> None:
    node_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).astype(np.float32)
    node = onnx.helper.make_node('Split', inputs=['input'], outputs=['output_1', 'output_2', 'output_3', 'output_4'], num_outputs=4)
    expected_outputs = [np.array([1.0, 2.0]).astype(np.float32), np.array([3.0, 4.0]).astype(np.float32), np.array([5.0, 6.0]).astype(np.float32), np.array([7.0]).astype(np.float32)]
    expect(node, inputs=[node_input], outputs=expected_outputs, name='test_split_1d_uneven_split_opset18')