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
def test_attr_string(self) -> None:
    attr = helper.make_attribute('str', b'test')
    self.assertEqual(attr.name, 'str')
    self.assertEqual(attr.s, b'test')
    checker.check_attribute(attr)
    attr = helper.make_attribute('str', 'test')
    self.assertEqual(attr.name, 'str')
    self.assertEqual(attr.s, b'test')
    checker.check_attribute(attr)
    attr = helper.make_attribute('str', 'test')
    self.assertEqual(attr.name, 'str')
    self.assertEqual(attr.s, b'test')
    checker.check_attribute(attr)
    attr = helper.make_attribute('str', '')
    self.assertEqual(attr.name, 'str')
    self.assertEqual(helper.get_attribute_value(attr), b'')
    checker.check_attribute(attr)