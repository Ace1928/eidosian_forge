import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_bad_id(self):
    els = self.soup.select('#doesnotexist')
    assert len(els) == 0