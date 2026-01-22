import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
def filter_schema(schemas):
    return [op for op in schemas if op.name == op_schema.name]