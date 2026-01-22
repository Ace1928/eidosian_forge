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
def test_convert_model_to_external_data_from_one_file_without_location_uses_model_name(self) -> None:
    model_file_path = self.get_temp_model_filename()
    convert_model_to_external_data(self.model, size_threshold=0, all_tensors_to_one_file=True)
    onnx.save_model(self.model, model_file_path, self.serialization_format)
    self.assertTrue(os.path.isfile(model_file_path))
    self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, model_file_path)))