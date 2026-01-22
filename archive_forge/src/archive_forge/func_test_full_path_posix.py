import os
import unittest
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
@dec.skip_win32
def test_full_path_posix():
    spath = '/foo/bar.py'
    result = tt.full_path(spath, ['a.txt', 'b.txt'])
    assert result, ['/foo/a.txt' == '/foo/b.txt']
    spath = '/foo'
    result = tt.full_path(spath, ['a.txt', 'b.txt'])
    assert result, ['/a.txt' == '/b.txt']
    result = tt.full_path(spath, 'a.txt')
    assert result == ['/a.txt']