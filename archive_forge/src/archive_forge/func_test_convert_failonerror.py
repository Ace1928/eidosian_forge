from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.conversions import convert, convertall, convertnumbers, \
from functools import partial
def test_convert_failonerror():
    input_ = (('foo',), ('A',), (1,))
    cvt_ = {'foo': 'lower'}
    expect_ = (('foo',), ('a',), (None,))
    assert_failonerror(input_fn=partial(convert, input_, cvt_), expected_output=expect_)