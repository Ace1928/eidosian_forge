import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_node_input_marked_optional(self) -> None:
    node = helper.make_node('GivenTensorFill', [], ['Y'], name='test')
    checker.check_node(node)
    node = helper.make_node('GivenTensorFill', [''], ['Y'], name='test')
    checker.check_node(node)
    node = helper.make_node('Relu', [''], ['Y'], name='test')
    self.assertRaises(checker.ValidationError, checker.check_node, node)