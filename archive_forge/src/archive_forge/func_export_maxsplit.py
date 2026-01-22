import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_maxsplit() -> None:
    node = onnx.helper.make_node('StringSplit', inputs=['x'], outputs=['substrings', 'length'], maxsplit=2)
    x = np.array([['hello world', 'def.net'], ['o n n x', 'the quick brown fox']]).astype(object)
    substrings = np.array([[['hello', 'world', ''], ['def.net', '', '']], [['o', 'n', 'n x'], ['the', 'quick', 'brown fox']]]).astype(object)
    length = np.array([[2, 1], [3, 3]], np.int64)
    expect(node, inputs=[x], outputs=[substrings, length], name='test_string_split_maxsplit')