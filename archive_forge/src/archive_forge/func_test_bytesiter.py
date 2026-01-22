import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_bytesiter():
    it = six.iterbytes(six.b('hi'))
    assert six.next(it) == ord('h')
    assert six.next(it) == ord('i')
    pytest.raises(StopIteration, six.next, it)