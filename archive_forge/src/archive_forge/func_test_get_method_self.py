import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_get_method_self():

    class X(object):

        def m(self):
            pass
    x = X()
    assert six.get_method_self(x.m) is x
    pytest.raises(AttributeError, six.get_method_self, 42)