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
def test_alternate_string_containers(self):

    class PString(NavigableString):
        pass

    class BString(NavigableString):
        pass
    soup = self.soup('<div>Hello.<p>Here is <b>some <i>bolded</i></b> text', string_containers={'b': BString, 'p': PString})
    assert isinstance(soup.div.contents[0], NavigableString)
    assert isinstance(soup.p.contents[0], PString)
    for s in soup.b.strings:
        assert isinstance(s, BString)
    assert [] == soup.string_container_stack