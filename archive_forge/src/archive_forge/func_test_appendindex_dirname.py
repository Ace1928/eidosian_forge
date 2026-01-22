from __future__ import absolute_import, print_function, division
import os
import tempfile
import pytest
from petl.test.helpers import ieq
import petl as etl
from petl.io.whoosh import fromtextindex, totextindex, appendtextindex, \
def test_appendindex_dirname():
    dirname = tempfile.mkdtemp()
    tbl = (('f0', 'f1', 'f2', 'f3', 'f4'), (u'AAA', 12, 4.3, True, datetime.datetime.now()), (u'BBB', 6, 3.4, False, datetime.datetime(1900, 1, 31)), (u'CCC', 42, 7.8, True, datetime.datetime(2100, 12, 25)))
    schema = Schema(f0=TEXT(stored=True), f1=NUMERIC(int, stored=True), f2=NUMERIC(float, stored=True), f3=BOOLEAN(stored=True), f4=DATETIME(stored=True))
    totextindex(tbl, dirname, schema=schema)
    appendtextindex(tbl, dirname)
    actual = fromtextindex(dirname)
    expect = tbl + tbl[1:]
    ieq(expect, actual)