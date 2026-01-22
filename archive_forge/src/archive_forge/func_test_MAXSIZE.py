import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_MAXSIZE():
    try:
        six.MAXSIZE.__index__()
    except AttributeError:
        pass
    pytest.raises((ValueError, OverflowError), operator.mul, [None], six.MAXSIZE + 1)