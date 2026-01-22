import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_custom_move_module(self):
    attr = six.MovedModule('spam', 'six', 'six')
    six.add_move(attr)
    six.remove_move('spam')
    assert not hasattr(six.moves, 'spam')
    attr = six.MovedModule('spam', 'six', 'six')
    six.add_move(attr)
    from six.moves import spam
    assert spam is six
    six.remove_move('spam')
    assert not hasattr(six.moves, 'spam')