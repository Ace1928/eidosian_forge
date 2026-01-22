import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_one_tag_many(self):
    els = self.soup.select('div')
    assert len(els) == 4
    for div in els:
        assert div.name == 'div'
    el = self.soup.select_one('div')
    assert 'main' == el['id']