import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_with_metaclass():

    class Meta(type):
        pass

    class X(six.with_metaclass(Meta)):
        pass
    assert type(X) is Meta
    assert issubclass(X, object)

    class Base(object):
        pass

    class X(six.with_metaclass(Meta, Base)):
        pass
    assert type(X) is Meta
    assert issubclass(X, Base)

    class Base2(object):
        pass

    class X(six.with_metaclass(Meta, Base, Base2)):
        pass
    assert type(X) is Meta
    assert issubclass(X, Base)
    assert issubclass(X, Base2)
    assert X.__mro__ == (X, Base, Base2, object)

    class X(six.with_metaclass(Meta)):
        pass

    class MetaSub(Meta):
        pass

    class Y(six.with_metaclass(MetaSub, X)):
        pass
    assert type(Y) is MetaSub
    assert Y.__mro__ == (Y, X, object)