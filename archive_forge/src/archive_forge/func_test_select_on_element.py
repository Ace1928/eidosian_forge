import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_select_on_element(self):
    inner = self.soup.find('div', id='main')
    selected = inner.select('div')
    self.assert_selects_ids(selected, ['inner', 'data1'])