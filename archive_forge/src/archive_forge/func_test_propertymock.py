import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_propertymock(self):
    p = patch('%s.SomeClass.one' % __name__, new_callable=PropertyMock)
    mock = p.start()
    try:
        SomeClass.one
        mock.assert_called_once_with()
        s = SomeClass()
        s.one
        mock.assert_called_with()
        self.assertEqual(mock.mock_calls, [call(), call()])
        s.one = 3
        self.assertEqual(mock.mock_calls, [call(), call(), call(3)])
    finally:
        p.stop()