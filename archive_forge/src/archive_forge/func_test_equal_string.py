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
def test_equal_string(self) -> None:
    self._logical_binary_op('Equal', TensorProto.STRING)
    self._logical_binary_op_with_broadcasting('Equal', TensorProto.STRING)