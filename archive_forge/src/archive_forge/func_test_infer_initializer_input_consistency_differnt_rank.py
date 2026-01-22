from __future__ import annotations
import itertools
import unittest
from typing import Any, Sequence
import numpy as np
import pytest
from parameterized import parameterized
import onnx.shape_inference
from onnx import (
from onnx.defs import (
from onnx.helper import (
from onnx.parser import parse_graph
def test_infer_initializer_input_consistency_differnt_rank(self) -> None:
    initializer_shape = (8, 7, 9)
    input_shape = (None, 7)
    original_model = self.prepare_input_initializer_tensors(initializer_shape, input_shape)
    self.assertRaises(onnx.shape_inference.InferenceError, onnx.shape_inference.infer_shapes, original_model, strict_mode=True)