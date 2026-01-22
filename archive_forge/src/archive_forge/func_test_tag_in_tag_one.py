import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_tag_in_tag_one(self):
    els = self.soup.select('div div')
    self.assert_selects('div div', ['inner', 'data1'])