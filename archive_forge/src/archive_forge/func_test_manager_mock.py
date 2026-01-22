import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
def test_manager_mock(self):

    class Foo(object):
        one = 'one'
        two = 'two'
    manager = Mock()
    p1 = patch.object(Foo, 'one')
    p2 = patch.object(Foo, 'two')
    mock_one = p1.start()
    self.addCleanup(p1.stop)
    mock_two = p2.start()
    self.addCleanup(p2.stop)
    manager.attach_mock(mock_one, 'one')
    manager.attach_mock(mock_two, 'two')
    Foo.two()
    Foo.one()
    self.assertEqual(manager.mock_calls, [call.two(), call.one()])