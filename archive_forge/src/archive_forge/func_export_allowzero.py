import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_allowzero() -> None:
    original_shape = [0, 3, 4]
    test_cases = {'allowzero_reordered': np.array([3, 4, 0], dtype=np.int64)}
    data = np.random.random_sample(original_shape).astype(np.float32)
    for test_name, shape in test_cases.items():
        node = onnx.helper.make_node('Reshape', inputs=['data', 'shape'], outputs=['reshaped'], allowzero=1)
        reshaped = reshape_reference_implementation(data, shape, allowzero=1)
        expect(node, inputs=[data, shape], outputs=[reshaped], name='test_reshape_' + test_name)