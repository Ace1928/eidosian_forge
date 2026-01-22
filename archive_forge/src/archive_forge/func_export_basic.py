import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_basic() -> None:
    node = onnx.helper.make_node('RegexFullMatch', inputs=['X'], outputs=['Y'], pattern='www\\.[\\w.-]+\\.\\bcom\\b')
    x = np.array(['www.google.com', 'www.facebook.com', 'www.bbc.co.uk']).astype(object)
    result = np.array([True, True, False])
    expect(node, inputs=[x], outputs=[result], name='test_regex_full_match_basic')