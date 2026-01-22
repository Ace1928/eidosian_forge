import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_propertymock_returnvalue(self):
    m = MagicMock()
    p = PropertyMock()
    type(m).foo = p
    returned = m.foo
    p.assert_called_once_with()
    self.assertIsInstance(returned, MagicMock)
    self.assertNotIsInstance(returned, PropertyMock)