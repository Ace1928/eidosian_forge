import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_monday_empty_output() -> None:
    input = np.array(['monday', 'monday']).astype(object)
    output = np.array(['']).astype(object)
    stopwords = ['monday']
    node = onnx.helper.make_node('StringNormalizer', inputs=['x'], outputs=['y'], case_change_action='UPPER', is_case_sensitive=1, stopwords=stopwords)
    expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_empty_output')