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
def test_url_warning_with_unicode_url(self):
    url = 'http://www.crummyunicode.com/'
    with warnings.catch_warnings(record=True) as warning_list:
        soup = BeautifulSoup(url, 'html.parser')
    warning = self._assert_warning(warning_list, MarkupResemblesLocatorWarning)
    assert 'looks more like a URL' in str(warning.message)
    assert url not in str(warning.message)