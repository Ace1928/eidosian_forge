import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_empty_string_delimiter() -> None:
    for delimiter, test_name in (('', 'test_string_split_empty_string_delimiter'), (None, 'test_string_split_no_delimiter')):
        node = onnx.helper.make_node('StringSplit', inputs=['x'], outputs=['substrings', 'length'], delimiter=delimiter, maxsplit=None)
        x = np.array(['hello world !', '  hello   world !', ' hello world   ! ']).astype(object)
        substrings = np.array([['hello', 'world', '!'], ['hello', 'world', '!'], ['hello', 'world', '!']]).astype(object)
        length = np.array([3, 3, 3], dtype=np.int64)
        expect(node, inputs=[x], outputs=[substrings, length], name=test_name)