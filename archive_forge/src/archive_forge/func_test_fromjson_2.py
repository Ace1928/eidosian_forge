from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import json
import pytest
from petl.test.helpers import ieq
from petl import fromjson, fromdicts, tojson, tojsonarrays
def test_fromjson_2():
    f = NamedTemporaryFile(delete=False, mode='w')
    data = '[{"foo": "a", "bar": 1}, {"foo": "b"}, {"foo": "c", "bar": 2, "baz": true}]'
    f.write(data)
    f.close()
    actual = fromjson(f.name, header=['bar', 'baz', 'foo'])
    expect = (('bar', 'baz', 'foo'), (1, None, 'a'), (None, None, 'b'), (2, True, 'c'))
    ieq(expect, actual)
    ieq(expect, actual)