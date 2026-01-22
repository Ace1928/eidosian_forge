from unittest import mock
import urllib
from glance.common import exception
from glance.common.scripts import utils as script_utils
import glance.tests.utils as test_utils
@mock.patch('oslo_utils.timeutils.StopWatch')
def test_iterator_iterates_granularly(self, mock_sw):
    items = ['1', '2', '3']
    callback = mock.MagicMock()
    mock_sw.return_value.expired.side_effect = [False, True, False]
    cb_iter = script_utils.CallbackIterator(iter(items), callback, min_interval=30)
    iter_items = list(cb_iter)
    self.assertEqual(items, iter_items)
    callback.assert_has_calls([mock.call(2, 2), mock.call(1, 3)])
    mock_sw.assert_called_once_with(30)
    mock_sw.return_value.start.assert_called_once_with()
    mock_sw.return_value.restart.assert_called_once_with()
    callback.reset_mock()
    cb_iter.close()
    callback.assert_not_called()