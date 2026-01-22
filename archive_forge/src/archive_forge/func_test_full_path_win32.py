import os
import unittest
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
@dec.skip_if_not_win32
def test_full_path_win32():
    spath = 'c:\\foo\\bar.py'
    result = tt.full_path(spath, ['a.txt', 'b.txt'])
    assert result, ['c:\\foo\\a.txt' == 'c:\\foo\\b.txt']
    spath = 'c:\\foo'
    result = tt.full_path(spath, ['a.txt', 'b.txt'])
    assert result, ['c:\\a.txt' == 'c:\\b.txt']
    result = tt.full_path(spath, 'a.txt')
    assert result == ['c:\\a.txt']