import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_monday_insensintive_upper_twodim() -> None:
    input = np.array(['Monday', 'tuesday', 'wednesday', 'Monday', 'tuesday', 'wednesday']).astype(object).reshape([1, 6])
    output = np.array(['TUESDAY', 'WEDNESDAY', 'TUESDAY', 'WEDNESDAY']).astype(object).reshape([1, 4])
    stopwords = ['monday']
    node = onnx.helper.make_node('StringNormalizer', inputs=['x'], outputs=['y'], case_change_action='UPPER', stopwords=stopwords)
    expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_insensintive_upper_twodim')