import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_error_opset_import_mismatch(self) -> None:
    """Tests that providing models with different operator set imported produces an error."""
    m1, m2 = (_load_model(M1_DEF), _load_model(M2_DEF))
    m1 = helper.make_model(m1.graph, producer_name='test', opset_imports=[helper.make_opsetid('', 10)])
    m2 = helper.make_model(m2.graph, producer_name='test', opset_imports=[helper.make_opsetid('', 15)])
    io_map = [('B00', 'B01'), ('B10', 'B11'), ('B20', 'B21')]
    self.assertRaises(ValueError, compose.merge_models, m1, m2, io_map)
    m1 = version_converter.convert_version(m1, 15)
    m3 = compose.merge_models(m1, m2, io_map=io_map)
    checker.check_model(m3)