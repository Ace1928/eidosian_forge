import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_raise_from():
    try:
        try:
            raise Exception('blah')
        except Exception:
            ctx = sys.exc_info()[1]
            f = Exception('foo')
            six.raise_from(f, None)
    except Exception:
        tp, val, tb = sys.exc_info()
    if sys.version_info[:2] > (3, 0):
        assert val.__cause__ is None
        assert val.__context__ is ctx
        assert val.__suppress_context__
    assert str(val) == 'foo'