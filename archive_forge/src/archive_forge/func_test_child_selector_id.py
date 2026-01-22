import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_child_selector_id(self):
    self.assert_selects('.s1 > a#s1a2 span', ['s1a2s1'])