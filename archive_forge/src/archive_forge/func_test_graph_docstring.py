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
def test_graph_docstring(self) -> None:
    graph = helper.make_graph([], 'my graph', [], [], None, 'my docs')
    self.assertEqual(graph.name, 'my graph')
    self.assertEqual(graph.doc_string, 'my docs')