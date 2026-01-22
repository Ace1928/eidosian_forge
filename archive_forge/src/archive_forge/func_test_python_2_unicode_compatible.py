import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_python_2_unicode_compatible():

    @six.python_2_unicode_compatible
    class MyTest(object):

        def __str__(self):
            return six.u('hello')

        def __bytes__(self):
            return six.b('hello')
    my_test = MyTest()
    if six.PY2:
        assert str(my_test) == six.b('hello')
        assert unicode(my_test) == six.u('hello')
    elif six.PY3:
        assert bytes(my_test) == six.b('hello')
        assert str(my_test) == six.u('hello')
    assert getattr(six.moves.builtins, 'bytes', str)(my_test) == six.b('hello')