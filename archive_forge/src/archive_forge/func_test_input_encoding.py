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
@pytest.mark.parametrize('encoding', ['ascii', 'utf-8', 'utf-16', 'utf-32'])
def test_input_encoding(self, tmp_path, encoding):
    f_path = tmp_path / f'input_with_{encoding}_encoding.f90'
    with f_path.open('w', encoding=encoding) as ff:
        ff.write('\n                     subroutine foo()\n                     end subroutine foo\n                     ')
    mod = crackfortran.crackfortran([str(f_path)])
    assert mod[0]['name'] == 'foo'