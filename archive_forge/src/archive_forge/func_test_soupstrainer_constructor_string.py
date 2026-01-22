from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_soupstrainer_constructor_string(self):
    with warnings.catch_warnings(record=True) as w:
        strainer = SoupStrainer(text='text')
        assert strainer.text == 'text'
        [warning] = w
        msg = str(warning.message)
        assert warning.filename == __file__
        assert msg == "The 'text' argument to the SoupStrainer constructor is deprecated. Use 'string' instead."