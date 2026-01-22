import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_integer_types():
    assert isinstance(1, six.integer_types)
    assert isinstance(-1, six.integer_types)
    assert isinstance(six.MAXSIZE + 23, six.integer_types)
    assert not isinstance(0.1, six.integer_types)