from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.maps import fieldmap, rowmap, rowmapmany
from functools import partial
def test_rowmap_failonerror():
    input_ = (('foo',), ('A',), (1,), ('B',))
    mapper = lambda r: [r[0].lower()]
    expect_ = (('foo',), ('a',), ('b',))
    assert_failonerror(input_fn=partial(rowmap, input_, mapper, header=('foo',)), expected_output=expect_)