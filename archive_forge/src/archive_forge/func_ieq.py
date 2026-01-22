from __future__ import absolute_import, print_function, division
import os
import sys
import pytest
from petl.compat import izip_longest
def ieq(expect, actual, cast=None):
    """Test when values of a iterable are equals for each row and column"""
    ie = iter(expect)
    ia = iter(actual)
    ir = 0
    for re, ra in izip_longest(ie, ia, fillvalue=None):
        if cast:
            ra = cast(ra)
        if re is None and ra is None:
            continue
        if type(re) in (int, float, bool, str):
            eq_(re, ra)
            continue
        _ieq_row(re, ra, ir)
        ir = ir + 1