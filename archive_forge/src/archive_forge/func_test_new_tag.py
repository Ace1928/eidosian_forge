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
def test_new_tag(self):
    soup = self.soup('')
    new_tag = soup.new_tag('foo', bar='baz', attrs={'name': 'a name'})
    assert isinstance(new_tag, Tag)
    assert 'foo' == new_tag.name
    assert dict(bar='baz', name='a name') == new_tag.attrs
    assert None == new_tag.parent