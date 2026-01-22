import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_recarray_from_repr(self):
    a = np.array([(1, 'ABC'), (2, 'DEF')], dtype=[('foo', int), ('bar', 'S4')])
    recordarr = np.rec.array(a)
    recarr = a.view(np.recarray)
    recordview = a.view(np.dtype((np.record, a.dtype)))
    recordarr_r = eval('numpy.' + repr(recordarr), {'numpy': np})
    recarr_r = eval('numpy.' + repr(recarr), {'numpy': np})
    recordview_r = eval('numpy.' + repr(recordview), {'numpy': np})
    assert_equal(type(recordarr_r), np.recarray)
    assert_equal(recordarr_r.dtype.type, np.record)
    assert_equal(recordarr, recordarr_r)
    assert_equal(type(recarr_r), np.recarray)
    assert_equal(recarr_r.dtype.type, np.record)
    assert_equal(recarr, recarr_r)
    assert_equal(type(recordview_r), np.ndarray)
    assert_equal(recordview.dtype.type, np.record)
    assert_equal(recordview, recordview_r)