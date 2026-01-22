import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_tag_in_tag_many(self):
    for selector in ('html div', 'html body div', 'body div'):
        self.assert_selects(selector, ['data1', 'main', 'inner', 'footer'])