import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
def test_deregister_schema_raises_error_when_opschema_does_not_exist(self):
    with self.assertRaises(onnx.defs.SchemaError):
        onnx.defs.deregister_schema(self.op_type, self.op_version, self.op_domain)