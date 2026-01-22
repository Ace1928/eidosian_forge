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
def test_unicode_in_unicode_out(self):
    soup_from_unicode = self.soup(self.unicode_data)
    assert soup_from_unicode.decode() == self.unicode_data
    assert soup_from_unicode.foo.string == 'Sacré bleu!'
    assert soup_from_unicode.original_encoding == None