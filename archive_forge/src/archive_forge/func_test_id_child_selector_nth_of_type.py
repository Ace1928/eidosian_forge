import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_id_child_selector_nth_of_type(self):
    self.assert_selects('#inner > p:nth-of-type(2)', ['p1'])