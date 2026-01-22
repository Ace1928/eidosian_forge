import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_signature_callable(self):

    class Callable(object):

        def __init__(self, x, y):
            pass

        def __call__(self, a):
            pass
    mock = create_autospec(Callable)
    mock(1, 2)
    mock.assert_called_once_with(1, 2)
    mock.assert_called_once_with(x=1, y=2)
    self.assertRaises(TypeError, mock, 'a')
    instance = mock(1, 2)
    self.assertRaises(TypeError, instance)
    instance(a='a')
    instance.assert_called_once_with('a')
    instance.assert_called_once_with(a='a')
    instance('a')
    instance.assert_called_with('a')
    instance.assert_called_with(a='a')
    mock = create_autospec(Callable(1, 2))
    mock(a='a')
    mock.assert_called_once_with(a='a')
    self.assertRaises(TypeError, mock)
    mock('a')
    mock.assert_called_with('a')