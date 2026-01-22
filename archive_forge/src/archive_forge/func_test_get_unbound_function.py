import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_get_unbound_function():

    class X(object):

        def m(self):
            pass
    assert six.get_unbound_function(X.m) is X.__dict__['m']