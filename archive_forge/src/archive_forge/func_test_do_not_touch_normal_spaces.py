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
def test_do_not_touch_normal_spaces(self):
    test_list = ['a ', ' a', 'a b c', "'abcdefghij'"]
    for i in test_list:
        assert markinnerspaces(i) == i