import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_attribute_endswith(self):
    self.assert_select_multiple(('[href$=".css"]', ['l1']), ('link[href$=".css"]', ['l1']), ('link[id$="1"]', ['l1']), ('[id$="1"]', ['data1', 'l1', 'p1', 'header1', 's1a1', 's2a1', 's1a2s1', 'dash1']), ('div[id$="1"]', ['data1']), ('[id$="noending"]', []))