import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_nostopwords_nochangecase() -> None:
    input = np.array(['monday', 'tuesday']).astype(object)
    output = input
    node = onnx.helper.make_node('StringNormalizer', inputs=['x'], outputs=['y'], is_case_sensitive=1)
    expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_nostopwords_nochangecase')