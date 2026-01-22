import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_select_one_returns_none_if_no_match(self):
    match = self.soup.select_one('nonexistenttag')
    assert None == match