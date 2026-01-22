from unittest import mock
import urllib
from glance.common import exception
from glance.common.scripts import utils as script_utils
import glance.tests.utils as test_utils
def test_iterator_iterates(self):
    items = ['1', '2', '', '3']
    callback = mock.MagicMock()
    cb_iter = script_utils.CallbackIterator(iter(items), callback)
    iter_items = list(cb_iter)
    callback.assert_has_calls([mock.call(1, 1), mock.call(1, 2), mock.call(1, 3)])
    self.assertEqual(items, iter_items)
    callback.reset_mock()
    cb_iter.close()
    callback.assert_not_called()