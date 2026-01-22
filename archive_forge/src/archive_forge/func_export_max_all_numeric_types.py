import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.backend.test.case.utils import all_numeric_dtypes
@staticmethod
def export_max_all_numeric_types() -> None:
    for op_dtype in all_numeric_dtypes:
        data_0 = np.array([3, 2, 1]).astype(op_dtype)
        data_1 = np.array([1, 4, 4]).astype(op_dtype)
        result = np.array([3, 4, 4]).astype(op_dtype)
        node = onnx.helper.make_node('Max', inputs=['data_0', 'data_1'], outputs=['result'])
        expect(node, inputs=[data_0, data_1], outputs=[result], name=f'test_max_{np.dtype(op_dtype).name}')