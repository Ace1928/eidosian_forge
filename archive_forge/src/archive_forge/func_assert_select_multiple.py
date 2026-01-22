import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def assert_select_multiple(self, *tests):
    for selector, expected_ids in tests:
        self.assert_selects(selector, expected_ids)