import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_model_supports_unicode_path(self):
    input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1])
    output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1])
    node = helper.make_node('Identity', ['input'], ['output'])
    graph = helper.make_graph([node], 'test', [input_tensor], [output_tensor])
    model = helper.make_model(graph, producer_name='test')
    with tempfile.TemporaryDirectory() as temp_dir:
        unicode_model_path = os.path.join(temp_dir, '模型モデル모델✨.onnx')
        onnx.save(model, unicode_model_path)
        checker.check_model(unicode_model_path, full_check=True)