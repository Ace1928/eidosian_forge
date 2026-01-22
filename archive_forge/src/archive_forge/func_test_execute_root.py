from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import privileged
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
@mock.patch('oslo_concurrency.processutils.execute')
def test_execute_root(self, mock_putils_exec):
    priv_rootwrap.execute_root('echo', 'foo', check_exit_code=0)
    mock_putils_exec.assert_called_once_with('echo', 'foo', check_exit_code=0, shell=False, run_as_root=False, delay_on_retry=False, on_completion=mock.ANY, on_execute=mock.ANY)
    self.assertRaises(TypeError, priv_rootwrap.execute_root, 'foo', shell=True)
    self.assertRaises(TypeError, priv_rootwrap.execute_root, 'foo', run_as_root=True)