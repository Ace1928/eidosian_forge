import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_wraps():

    def f(g):

        @six.wraps(g)
        def w():
            return 42
        return w

    def k():
        pass
    original_k = k
    k = f(f(k))
    assert hasattr(k, '__wrapped__')
    k = k.__wrapped__
    assert hasattr(k, '__wrapped__')
    k = k.__wrapped__
    assert k is original_k
    assert not hasattr(k, '__wrapped__')

    def f(g, assign, update):

        def w():
            return 42
        w.glue = {'foo': 'bar'}
        w.xyzzy = {'qux': 'quux'}
        return six.wraps(g, assign, update)(w)
    k.glue = {'melon': 'egg'}
    k.turnip = 43
    k = f(k, ['turnip', 'baz'], ['glue', 'xyzzy'])
    assert k.__name__ == 'w'
    assert k.turnip == 43
    assert not hasattr(k, 'baz')
    assert k.glue == {'melon': 'egg', 'foo': 'bar'}
    assert k.xyzzy == {'qux': 'quux'}