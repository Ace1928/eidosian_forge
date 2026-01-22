from __future__ import absolute_import, print_function, division
import pytest
from petl.compat import next
from petl.errors import ArgumentError
from petl.test.helpers import ieq, eq_
from petl.transform.regex import capture, split, search, searchcomplement, splitdown
from petl.transform.basics import TransformError
def test_search_unicode():
    tbl = ((u'name', u'id'), (u'Արամ Խաչատրյան', 1), (u'Johann Strauß', 2), (u'Вагиф Сәмәдоғлу', 3), (u'章子怡', 4))
    actual = search(tbl, u'.Խա.')
    expect = ((u'name', u'id'), (u'Արամ Խաչատրյան', 1))
    ieq(expect, actual)
    ieq(expect, actual)