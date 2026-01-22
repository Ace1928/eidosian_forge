import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_callable():

    class X:

        def __call__(self):
            pass

        def method(self):
            pass
    assert six.callable(X)
    assert six.callable(X())
    assert six.callable(test_callable)
    assert six.callable(hasattr)
    assert six.callable(X.method)
    assert six.callable(X().method)
    assert not six.callable(4)
    assert not six.callable('string')