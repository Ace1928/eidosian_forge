import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_attribute_tilde(self):
    self.assert_select_multiple(('p[class~="class1"]', ['pmulti']), ('p[class~="class2"]', ['pmulti']), ('p[class~="class3"]', ['pmulti']), ('[class~="class1"]', ['pmulti']), ('[class~="class2"]', ['pmulti']), ('[class~="class3"]', ['pmulti']), ('a[rel~="friend"]', ['bob']), ('a[rel~="met"]', ['bob']), ('[rel~="friend"]', ['bob']), ('[rel~="met"]', ['bob']))