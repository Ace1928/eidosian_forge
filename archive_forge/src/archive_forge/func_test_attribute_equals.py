import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_attribute_equals(self):
    self.assert_select_multiple(('p[class="onep"]', ['p1']), ('p[id="p1"]', ['p1']), ('[class="onep"]', ['p1']), ('[id="p1"]', ['p1']), ('link[rel="stylesheet"]', ['l1']), ('link[type="text/css"]', ['l1']), ('link[href="blah.css"]', ['l1']), ('link[href="no-blah.css"]', []), ('[rel="stylesheet"]', ['l1']), ('[type="text/css"]', ['l1']), ('[href="blah.css"]', ['l1']), ('[href="no-blah.css"]', []), ('p[href="no-blah.css"]', []), ('[href="no-blah.css"]', []))