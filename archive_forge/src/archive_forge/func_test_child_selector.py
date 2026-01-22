import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_child_selector(self):
    self.assert_selects('.s1 > a', ['s1a1', 's1a2'])
    self.assert_selects('.s1 > a span', ['s1a2s1'])