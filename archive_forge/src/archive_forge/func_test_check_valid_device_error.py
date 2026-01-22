import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
def test_check_valid_device_error(self):
    mock_execute = mock.Mock()
    p_exception = utils.processutils.ProcessExecutionError
    mock_execute._execute.side_effect = p_exception
    fake_path = '/dev/fake'
    is_valid = utils.check_valid_device(mock_execute, fake_path)
    self.assertEqual(False, is_valid)
    mock_execute._execute.assert_called_once_with('dd', 'if=/dev/fake', 'of=/dev/null', 'count=1', run_as_root=True, root_helper=mock_execute._root_helper)