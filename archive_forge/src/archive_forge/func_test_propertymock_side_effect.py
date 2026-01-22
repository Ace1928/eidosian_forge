import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_propertymock_side_effect(self):
    m = MagicMock()
    p = PropertyMock(side_effect=ValueError)
    type(m).foo = p
    with self.assertRaises(ValueError):
        m.foo
    p.assert_called_once_with()