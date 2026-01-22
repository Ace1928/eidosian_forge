import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_advance_iterator():
    assert six.next is six.advance_iterator
    l = [1, 2]
    it = iter(l)
    assert six.next(it) == 1
    assert six.next(it) == 2
    pytest.raises(StopIteration, six.next, it)
    pytest.raises(StopIteration, six.next, it)