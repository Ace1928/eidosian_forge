import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_add_prefix_all(self) -> None:
    """Tests prefixing all names in the graph"""
    self._test_add_prefix(True, True, True, True, True, True)