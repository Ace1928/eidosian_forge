from pdb import set_trace
import logging
import os
import pickle
import pytest
import sys
import tempfile
from bs4 import (
from bs4.builder import (
from bs4.element import (
from . import (
import warnings
def test_parseOnlyThese_renamed_to_parse_only(self):
    with warnings.catch_warnings(record=True) as w:
        soup = BeautifulSoup('<a><b></b></a>', 'html.parser', parseOnlyThese=SoupStrainer('b'))
    warning = self._assert_warning(w, DeprecationWarning)
    msg = str(warning.message)
    assert 'parseOnlyThese' in msg
    assert 'parse_only' in msg
    assert b'<b></b>' == soup.encode()