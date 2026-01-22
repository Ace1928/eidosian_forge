import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_nth_of_type(self):
    els = self.soup.select('div#inner p:nth-of-type(1)')
    assert len(els) == 1
    assert els[0].string == 'Some text'
    els = self.soup.select('div#inner p:nth-of-type(3)')
    assert len(els) == 1
    assert els[0].string == 'Another'
    els = self.soup.select('div#inner p:nth-of-type(4)')
    assert len(els) == 0
    els = self.soup.select('div p:nth-of-type(0)')
    assert len(els) == 0