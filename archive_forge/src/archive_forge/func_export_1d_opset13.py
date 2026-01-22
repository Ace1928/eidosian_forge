import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_1d_opset13() -> None:
    node_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)
    node = onnx.helper.make_node('Split', inputs=['input'], outputs=['output_1', 'output_2', 'output_3'], axis=0)
    expected_outputs = [np.array([1.0, 2.0]).astype(np.float32), np.array([3.0, 4.0]).astype(np.float32), np.array([5.0, 6.0]).astype(np.float32)]
    expect(node, inputs=[node_input], outputs=expected_outputs, name='test_split_equal_parts_1d_opset13', opset_imports=[onnx.helper.make_opsetid('', 13)])
    split = np.array([2, 4]).astype(np.int64)
    node = onnx.helper.make_node('Split', inputs=['input', 'split'], outputs=['output_1', 'output_2'], axis=0)
    expected_outputs = [np.array([1.0, 2.0]).astype(np.float32), np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32)]
    expect(node, inputs=[node_input, split], outputs=expected_outputs, name='test_split_variable_parts_1d_opset13', opset_imports=[onnx.helper.make_opsetid('', 13)])