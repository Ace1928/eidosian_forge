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
def test_save_model_without_loading_external_data(self) -> None:
    model_file_path = self.get_temp_model_filename()
    onnx.save_model(self.model, model_file_path, self.serialization_format, save_as_external_data=True, location=None, size_threshold=0, convert_attribute=False)
    model = onnx.load_model(model_file_path, self.serialization_format, load_external_data=False)
    onnx.save_model(model, model_file_path, self.serialization_format, save_as_external_data=True, location=None, size_threshold=0, convert_attribute=False)
    model = onnx.load_model(model_file_path, self.serialization_format)
    initializer_tensor = model.graph.initializer[0]
    self.assertTrue(initializer_tensor.HasField('data_location'))
    np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)
    attribute_tensor = model.graph.node[0].attribute[0].t
    self.assertFalse(attribute_tensor.HasField('data_location'))
    np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)