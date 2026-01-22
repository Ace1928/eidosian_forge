import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_tag_no_match(self):
    assert len(self.soup.select('del')) == 0