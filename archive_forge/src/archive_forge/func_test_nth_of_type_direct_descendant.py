import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_nth_of_type_direct_descendant(self):
    els = self.soup.select('div#inner > p:nth-of-type(1)')
    assert len(els) == 1
    assert els[0].string == 'Some text'