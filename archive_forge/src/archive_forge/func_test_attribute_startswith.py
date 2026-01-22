import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_attribute_startswith(self):
    self.assert_select_multiple(('[rel^="style"]', ['l1']), ('link[rel^="style"]', ['l1']), ('notlink[rel^="notstyle"]', []), ('[rel^="notstyle"]', []), ('link[rel^="notstyle"]', []), ('link[href^="bla"]', ['l1']), ('a[href^="http://"]', ['bob', 'me']), ('[href^="http://"]', ['bob', 'me']), ('[id^="p"]', ['pmulti', 'p1']), ('[id^="m"]', ['me', 'main']), ('div[id^="m"]', ['main']), ('a[id^="m"]', ['me']), ('div[data-tag^="dashed"]', ['data1']))