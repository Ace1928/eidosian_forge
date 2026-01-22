import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
def test_register_schema_raises_error_when_registering_a_schema_twice(self):
    op_schema = defs.OpSchema(self.op_type, self.op_domain, self.op_version)
    onnx.defs.register_schema(op_schema)
    with self.assertRaises(onnx.defs.SchemaError):
        onnx.defs.register_schema(op_schema)