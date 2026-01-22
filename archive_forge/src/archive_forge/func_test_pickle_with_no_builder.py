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
def test_pickle_with_no_builder(self):
    soup = self.soup('some markup')
    soup.builder = None
    pickled = pickle.dumps(soup)
    unpickled = pickle.loads(pickled)
    assert 'some markup' == unpickled.string