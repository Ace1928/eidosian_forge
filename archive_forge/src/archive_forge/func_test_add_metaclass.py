import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_add_metaclass():

    class Meta(type):
        pass

    class X:
        """success"""
    X = six.add_metaclass(Meta)(X)
    assert type(X) is Meta
    assert issubclass(X, object)
    assert X.__module__ == __name__
    assert X.__doc__ == 'success'

    class Base(object):
        pass

    class X(Base):
        pass
    X = six.add_metaclass(Meta)(X)
    assert type(X) is Meta
    assert issubclass(X, Base)

    class Base2(object):
        pass

    class X(Base, Base2):
        pass
    X = six.add_metaclass(Meta)(X)
    assert type(X) is Meta
    assert issubclass(X, Base)
    assert issubclass(X, Base2)

    class Meta1(type):
        m1 = 'm1'

    class Meta2(Meta1):
        m2 = 'm2'

    class Base:
        b = 'b'
    Base = six.add_metaclass(Meta1)(Base)

    class X(Base):
        x = 'x'
    X = six.add_metaclass(Meta2)(X)
    assert type(X) is Meta2
    assert issubclass(X, Base)
    assert type(Base) is Meta1
    assert '__dict__' not in vars(X)
    instance = X()
    instance.attr = 'test'
    assert vars(instance) == {'attr': 'test'}
    assert instance.b == Base.b
    assert instance.x == X.x

    class MySlots(object):
        __slots__ = ['a', 'b']
    MySlots = six.add_metaclass(Meta1)(MySlots)
    assert MySlots.__slots__ == ['a', 'b']
    instance = MySlots()
    instance.a = 'foo'
    pytest.raises(AttributeError, setattr, instance, 'c', 'baz')

    class MyStringSlots(object):
        __slots__ = 'ab'
    MyStringSlots = six.add_metaclass(Meta1)(MyStringSlots)
    assert MyStringSlots.__slots__ == 'ab'
    instance = MyStringSlots()
    instance.ab = 'foo'
    pytest.raises(AttributeError, setattr, instance, 'a', 'baz')
    pytest.raises(AttributeError, setattr, instance, 'b', 'baz')

    class MySlotsWeakref(object):
        __slots__ = ('__weakref__',)
    MySlotsWeakref = six.add_metaclass(Meta)(MySlotsWeakref)
    assert type(MySlotsWeakref) is Meta