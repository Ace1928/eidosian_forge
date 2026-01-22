import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_select_dashed_matches_find_all(self):
    assert self.soup.select('custom-dashed-tag') == self.soup.find_all('custom-dashed-tag')