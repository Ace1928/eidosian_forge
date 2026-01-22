import os
from unittest import mock
from cliff.tests import base
from cliff import utils
@mock.patch('cliff.utils.os')
def test_get_terminal_size(self, mock_os):
    ts = os.terminal_size((10, 5))
    mock_os.get_terminal_size.return_value = ts
    width = utils.terminal_width()
    self.assertEqual(10, width)
    mock_os.get_terminal_size.side_effect = OSError()
    width = utils.terminal_width()
    self.assertIs(None, width)