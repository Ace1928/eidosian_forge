from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.conversions import convert, convertall, convertnumbers, \
from functools import partial
def test_convert_translate():
    table = (('foo', 'bar'), ('M', 12), ('F', 34), ('-', 56))
    trans = {'M': 'male', 'F': 'female'}
    result = convert(table, 'foo', trans)
    expectation = (('foo', 'bar'), ('male', 12), ('female', 34), ('-', 56))
    ieq(expectation, result)