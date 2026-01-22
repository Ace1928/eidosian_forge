from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import privileged
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
@mock.patch.object(priv_rootwrap.execute_root.privsep_entrypoint, 'client_mode', False)
@mock.patch.object(priv_rootwrap, 'custom_execute')
def test_execute_as_root(self, exec_mock):
    res = priv_rootwrap.execute(mock.sentinel.cmds, run_as_root=True, root_helper=mock.sentinel.root_helper, keyword_arg=mock.sentinel.kwarg)
    self.assertEqual(exec_mock.return_value, res)
    exec_mock.assert_called_once_with(mock.sentinel.cmds, shell=False, run_as_root=False, keyword_arg=mock.sentinel.kwarg)