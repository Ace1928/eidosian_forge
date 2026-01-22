import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_consecutive_delimiters() -> None:
    node = onnx.helper.make_node('StringSplit', inputs=['x'], outputs=['substrings', 'length'], delimiter='-', maxsplit=None)
    x = np.array(['o-n-n--x-', 'o-n----nx']).astype(object)
    substrings = np.array([['o', 'n', 'n', '', 'x', ''], ['o', 'n', '', '', '', 'nx']]).astype(object)
    length = np.array([6, 6], dtype=np.int64)
    expect(node, inputs=[x], outputs=[substrings, length], name='test_string_split_consecutive_delimiters')