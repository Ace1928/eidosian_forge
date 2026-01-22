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
def test_new_string_creates_navigablestring(self):
    soup = self.soup('')
    s = soup.new_string('foo')
    assert 'foo' == s
    assert isinstance(s, NavigableString)