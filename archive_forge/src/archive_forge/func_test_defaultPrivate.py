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
def test_defaultPrivate(self):
    fpath = util.getpath('tests', 'src', 'crackfortran', 'privatemod.f90')
    mod = crackfortran.crackfortran([str(fpath)])
    assert len(mod) == 1
    mod = mod[0]
    assert 'private' in mod['vars']['a']['attrspec']
    assert 'public' not in mod['vars']['a']['attrspec']
    assert 'private' in mod['vars']['b']['attrspec']
    assert 'public' not in mod['vars']['b']['attrspec']
    assert 'private' not in mod['vars']['seta']['attrspec']
    assert 'public' in mod['vars']['seta']['attrspec']