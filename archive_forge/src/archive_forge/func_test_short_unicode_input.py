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
def test_short_unicode_input(self):
    data = '<h1>éé</h1>'
    soup = self.soup(data)
    assert 'éé' == soup.h1.string