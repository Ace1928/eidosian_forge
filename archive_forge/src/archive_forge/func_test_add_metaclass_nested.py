import operator
import sys
import types
import unittest
import abc
import pytest
import six
@pytest.mark.skipif('sys.version_info[:2] < (3, 3)')
def test_add_metaclass_nested():

    class Meta(type):
        pass

    class A:

        class B:
            pass
    expected = 'test_add_metaclass_nested.<locals>.A.B'
    assert A.B.__qualname__ == expected

    class A:

        @six.add_metaclass(Meta)
        class B:
            pass
    assert A.B.__qualname__ == expected