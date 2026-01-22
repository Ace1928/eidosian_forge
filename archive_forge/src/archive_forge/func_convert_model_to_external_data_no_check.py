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
def convert_model_to_external_data_no_check(model: ModelProto, location: str):
    for tensor in model.graph.initializer:
        if tensor.HasField('raw_data'):
            set_external_data(tensor, location)