import typing
import unittest
import onnx
import onnx.parser
import onnx.shape_inference
def test_mi_constant(self):
    model = '\n            <\n                ir_version: 7,\n                opset_import: [ "" : 17]\n            >\n            mymodel (float[4, 8, 16] x) => (y) {\n                shape = Constant<value_ints=[8,4,16]>()\n                y = Reshape(x, shape)\n            }\n            '
    self._check_shape(model, [8, 4, 16])