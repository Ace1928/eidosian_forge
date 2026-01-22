from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.maps import fieldmap, rowmap, rowmapmany
from functools import partial
def test_fieldmap_failonerror():
    input_ = (('foo',), ('A',), (1,))
    mapper_ = {'bar': ('foo', lambda v: v.lower())}
    expect_ = (('bar',), ('a',), (None,))
    assert_failonerror(input_fn=partial(fieldmap, input_, mapper_), expected_output=expect_)