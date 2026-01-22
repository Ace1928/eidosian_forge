from unittest import mock
import urllib
from glance.common import exception
from glance.common.scripts import utils as script_utils
import glance.tests.utils as test_utils
def test_proxy_close(self):
    callback = mock.MagicMock()
    source = mock.MagicMock()
    del source.close
    script_utils.CallbackIterator(source, callback).close()
    source = mock.MagicMock()
    source.close.return_value = 'foo'
    script_utils.CallbackIterator(source, callback).close()
    source.close.assert_called_once_with()
    callback.assert_not_called()