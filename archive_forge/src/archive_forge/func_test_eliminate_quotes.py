import pytest
from numpy.f2py.symbolic import (
from . import util
def test_eliminate_quotes(self):

    def worker(s):
        r, d = eliminate_quotes(s)
        s1 = insert_quotes(r, d)
        assert s1 == s
    for kind in ['', 'mykind_']:
        worker(kind + '"1234" // "ABCD"')
        worker(kind + '"1234" // ' + kind + '"ABCD"')
        worker(kind + '"1234" // \'ABCD\'')
        worker(kind + '"1234" // ' + kind + "'ABCD'")
        worker(kind + '"1\\"2\'AB\'34"')
        worker('a = ' + kind + '\'1\\\'2"AB"34\'')