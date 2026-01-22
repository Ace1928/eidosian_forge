from __future__ import absolute_import, print_function, division
import tempfile
import pytest
from petl.test.helpers import ieq, eq_
from petl.io.bcolz import frombcolz, tobcolz, appendbcolz
def test_appendbcolz():
    t = [('foo', 'bar', 'baz'), ('apples', 1, 2.5), ('oranges', 3, 4.4), ('pears', 7, 0.1)]
    ctbl = tobcolz(t)
    appendbcolz(t, ctbl)
    eq_(t[0], tuple(ctbl.names))
    ieq(t[1:] + t[1:], (tuple(r) for r in ctbl.iter()))
    rootdir = tempfile.mkdtemp()
    tobcolz(t, rootdir=rootdir)
    appendbcolz(t, rootdir)
    ctbl = bcolz.open(rootdir, mode='r')
    eq_(t[0], tuple(ctbl.names))
    ieq(t[1:] + t[1:], (tuple(r) for r in ctbl.iter()))