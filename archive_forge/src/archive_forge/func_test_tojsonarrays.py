from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import json
import pytest
from petl.test.helpers import ieq
from petl import fromjson, fromdicts, tojson, tojsonarrays
def test_tojsonarrays():
    table = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2))
    f = NamedTemporaryFile(delete=False, mode='r')
    tojsonarrays(table, f.name)
    result = json.load(f)
    assert len(result) == 3
    assert result[0][0] == 'a'
    assert result[0][1] == 1
    assert result[1][0] == 'b'
    assert result[1][1] == 2
    assert result[2][0] == 'c'
    assert result[2][1] == 2