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
def test_attach_mock_return_value(self):
    classes = (Mock, MagicMock, NonCallableMagicMock, NonCallableMock)
    for Klass in (Mock, MagicMock):
        for Klass2 in classes:
            m = Klass()
            m2 = Klass2(name='foo')
            m.attach_mock(m2, 'return_value')
            self.assertIs(m(), m2)
            self.assertIn("name='mock()'", repr(m2))
            m2.foo()
            self.assertEqual(m.mock_calls, call().foo().call_list())