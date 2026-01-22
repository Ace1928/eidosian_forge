import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_dashed_tag_text(self):
    assert self.soup.select('body > custom-dashed-tag')[0].text == 'Hello there.'