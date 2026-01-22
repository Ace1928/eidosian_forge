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
def test_moduleOperators(self, tmp_path):
    fpath = util.getpath('tests', 'src', 'crackfortran', 'operators.f90')
    mod = crackfortran.crackfortran([str(fpath)])
    assert len(mod) == 1
    mod = mod[0]
    assert 'body' in mod and len(mod['body']) == 9
    assert mod['body'][1]['name'] == 'operator(.item.)'
    assert 'implementedby' in mod['body'][1]
    assert mod['body'][1]['implementedby'] == ['item_int', 'item_real']
    assert mod['body'][2]['name'] == 'operator(==)'
    assert 'implementedby' in mod['body'][2]
    assert mod['body'][2]['implementedby'] == ['items_are_equal']
    assert mod['body'][3]['name'] == 'assignment(=)'
    assert 'implementedby' in mod['body'][3]
    assert mod['body'][3]['implementedby'] == ['get_int', 'get_real']