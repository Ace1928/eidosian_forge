from __future__ import annotations
import itertools
import os
import pathlib
import tempfile
import unittest
import uuid
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import ModelProto, TensorProto, checker, helper, shape_inference
from onnx.external_data_helper import (
from onnx.numpy_helper import from_array, to_array
def test_convert_model_to_external_data_converts_attribute_values(self) -> None:
    model_file_path = self.get_temp_model_filename()
    convert_model_to_external_data(self.model, size_threshold=0, convert_attribute=True)
    onnx.save_model(self.model, model_file_path, self.serialization_format)
    model = onnx.load_model(model_file_path, self.serialization_format)
    initializer_tensor = model.graph.initializer[0]
    np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)
    self.assertTrue(initializer_tensor.HasField('data_location'))
    attribute_tensor = model.graph.node[0].attribute[0].t
    np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)
    self.assertTrue(attribute_tensor.HasField('data_location'))