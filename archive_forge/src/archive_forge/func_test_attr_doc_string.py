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
def test_attr_doc_string(self) -> None:
    node_def = helper.make_node('Relu', ['X'], ['Y'], name='test', doc_string='doc')
    self.assertEqual(node_def.doc_string, 'doc')