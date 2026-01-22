import importlib
import codecs
import time
import unicodedata
import pytest
import numpy as np
from numpy.f2py.crackfortran import markinnerspaces, nameargspattern
from . import util
from numpy.f2py import crackfortran
import textwrap
import contextlib
import io
def test_multiple_relevant_spaces(self):
    assert markinnerspaces("a 'b c' 'd e'") == "a 'b@_@c' 'd@_@e'"
    assert markinnerspaces('a "b c" "d e"') == 'a "b@_@c" "d@_@e"'