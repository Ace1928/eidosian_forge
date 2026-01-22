import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_print_exceptions():
    pytest.raises(TypeError, six.print_, x=3)
    pytest.raises(TypeError, six.print_, end=3)
    pytest.raises(TypeError, six.print_, sep=42)