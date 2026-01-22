import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_multiple_select_duplicated(self):
    self.assert_selects('x, x', ['xid'])