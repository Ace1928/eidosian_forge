import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_reshape() -> None:
    original_shape = [2, 3, 4]
    test_cases = {'reordered_all_dims': np.array([4, 2, 3], dtype=np.int64), 'reordered_last_dims': np.array([2, 4, 3], dtype=np.int64), 'reduced_dims': np.array([2, 12], dtype=np.int64), 'extended_dims': np.array([2, 3, 2, 2], dtype=np.int64), 'one_dim': np.array([24], dtype=np.int64), 'negative_dim': np.array([2, -1, 2], dtype=np.int64), 'negative_extended_dims': np.array([-1, 2, 3, 4], dtype=np.int64), 'zero_dim': np.array([2, 0, 4, 1], dtype=np.int64), 'zero_and_negative_dim': np.array([2, 0, 1, -1], dtype=np.int64)}
    data = np.random.random_sample(original_shape).astype(np.float32)
    for test_name, shape in test_cases.items():
        node = onnx.helper.make_node('Reshape', inputs=['data', 'shape'], outputs=['reshaped'])
        reshaped = reshape_reference_implementation(data, shape)
        expect(node, inputs=[data, shape], outputs=[reshaped], name='test_reshape_' + test_name)