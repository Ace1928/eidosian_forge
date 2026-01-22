import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_closest(self):
    inner = self.soup.find('div', id='inner')
    closest = inner.css.closest('div[id=main]')
    assert closest == self.soup.find('div', id='main')