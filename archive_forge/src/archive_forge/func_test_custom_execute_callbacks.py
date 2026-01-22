from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import privileged
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_custom_execute_callbacks(self):
    """Confirm execute callbacks are called on execute."""
    on_execute = mock.Mock()
    on_completion = mock.Mock()
    msg = 'hola'
    out, err = priv_rootwrap.custom_execute('echo', msg, on_execute=on_execute, on_completion=on_completion)
    self.assertEqual(msg + '\n', out)
    self.assertEqual('', err)
    on_execute.assert_called_once_with(mock.ANY)
    proc = on_execute.call_args[0][0]
    on_completion.assert_called_once_with(proc)