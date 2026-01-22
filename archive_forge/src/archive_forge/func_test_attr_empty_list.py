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
def test_attr_empty_list(self) -> None:
    attr = helper.make_attribute('empty', [], attr_type=AttributeProto.STRINGS)
    self.assertEqual(attr.type, AttributeProto.STRINGS)
    self.assertEqual(len(attr.strings), 0)
    self.assertRaises(ValueError, helper.make_attribute, 'empty', [])