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
@pytest.mark.parametrize('extension', ['markuphtml', 'markup.com', '', 'markup.js'])
def test_resembles_filename_no_warning(self, extension):
    with warnings.catch_warnings(record=True) as w:
        soup = self.soup('markup' + extension)
    assert [] == w