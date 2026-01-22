import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
@unittest.skipIf(six.PY3, 'Unicode is properly handled with Python 3')
def test_call_list_unicode(self):
    mock = Mock()

    class NonAsciiRepr(object):

        def __repr__(self):
            return 'é'
    mock(**{unicode('a'): NonAsciiRepr()})
    self.assertEqual(str(mock.mock_calls), '[call(a=é)]')