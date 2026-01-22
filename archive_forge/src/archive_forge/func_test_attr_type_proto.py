import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
def test_attr_type_proto(self) -> None:
    type_proto = TypeProto()
    attr = helper.make_attribute('type_proto', type_proto)
    self.assertEqual(attr.name, 'type_proto')
    self.assertEqual(attr.tp, type_proto)
    self.assertEqual(attr.type, AttributeProto.TYPE_PROTO)
    types = [TypeProto(), TypeProto()]
    attr = helper.make_attribute('type_protos', types)
    self.assertEqual(attr.name, 'type_protos')
    self.assertEqual(list(attr.type_protos), types)
    self.assertEqual(attr.type, AttributeProto.TYPE_PROTOS)