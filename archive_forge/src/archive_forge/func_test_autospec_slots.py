import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_autospec_slots(self):

    class Foo(object):
        __slots__ = ['a']
    foo = create_autospec(Foo)
    mock_slot = foo.a
    mock_slot(1, 2, 3)
    mock_slot.abc(4, 5, 6)
    mock_slot.assert_called_once_with(1, 2, 3)
    mock_slot.abc.assert_called_once_with(4, 5, 6)