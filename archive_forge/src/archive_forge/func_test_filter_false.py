import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_filter_false():
    from six.moves import filterfalse
    f = filterfalse(lambda x: x % 3, range(10))
    assert six.advance_iterator(f) == 0
    assert six.advance_iterator(f) == 3
    assert six.advance_iterator(f) == 6