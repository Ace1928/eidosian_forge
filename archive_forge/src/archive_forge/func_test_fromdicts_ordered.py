from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import json
import pytest
from petl.test.helpers import ieq
from petl import fromjson, fromdicts, tojson, tojsonarrays
def test_fromdicts_ordered():
    data = [OrderedDict([('foo', 'a'), ('bar', 1)]), OrderedDict([('foo', 'b')]), OrderedDict([('foo', 'c'), ('bar', 2), ('baz', True)])]
    actual = fromdicts(data)
    expect = (('foo', 'bar', 'baz'), ('a', 1, None), ('b', None, None), ('c', 2, True))
    ieq(expect, actual)