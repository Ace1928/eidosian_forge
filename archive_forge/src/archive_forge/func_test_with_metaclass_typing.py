import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_with_metaclass_typing():
    try:
        import typing
    except ImportError:
        pytest.skip('typing module required')

    class Meta(type):
        pass
    if sys.version_info[:2] < (3, 7):

        class Meta(Meta, typing.GenericMeta):
            pass
    T = typing.TypeVar('T')

    class G(six.with_metaclass(Meta, typing.Generic[T])):
        pass

    class GA(six.with_metaclass(abc.ABCMeta, typing.Generic[T])):
        pass
    assert isinstance(G, Meta)
    assert isinstance(GA, abc.ABCMeta)
    assert G[int] is not G[G[int]]
    assert GA[int] is not GA[GA[int]]
    assert G.__bases__ == (typing.Generic,)
    assert G.__orig_bases__ == (typing.Generic[T],)