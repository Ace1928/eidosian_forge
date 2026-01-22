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
def test_parse_with_soupstrainer(self):
    markup = 'No<b>Yes</b><a>No<b>Yes <c>Yes</c></b>'
    strainer = SoupStrainer('b')
    soup = self.soup(markup, parse_only=strainer)
    assert soup.encode() == b'<b>Yes</b><b>Yes <c>Yes</c></b>'