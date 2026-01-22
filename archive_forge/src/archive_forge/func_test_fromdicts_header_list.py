from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import json
import pytest
from petl.test.helpers import ieq
from petl import fromjson, fromdicts, tojson, tojsonarrays
def test_fromdicts_header_list():
    data = [OrderedDict([('foo', 'a'), ('bar', 1)]), OrderedDict([('foo', 'b'), ('bar', 2)]), OrderedDict([('foo', 'c'), ('bar', 2)])]
    actual = fromdicts(data)
    header = actual.header()
    assert header == ('foo', 'bar')
    expect = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2))
    ieq(expect, actual)
    ieq(expect, actual)