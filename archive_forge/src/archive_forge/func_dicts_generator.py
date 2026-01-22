from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import json
import pytest
from petl.test.helpers import ieq
from petl import fromjson, fromdicts, tojson, tojsonarrays
@pytest.fixture
def dicts_generator():

    def generator():
        yield OrderedDict([('foo', 'a'), ('bar', 1)])
        yield OrderedDict([('foo', 'b'), ('bar', 2)])
        yield OrderedDict([('foo', 'c'), ('bar', 2)])
    return generator()