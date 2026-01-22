import threading
from unittest import mock
from oslo_concurrency import processutils as putils
from oslo_context import context as context_utils
from os_brick import executor as brick_executor
from os_brick.privileged import rootwrap
from os_brick.tests import base
@mock.patch('sys.stdin', encoding='UTF-8')
@mock.patch('os_brick.executor.priv_rootwrap.execute')
def test_execute_non_safe_bytes_exception(self, execute_mock, stdin_mock):
    execute_mock.side_effect = putils.ProcessExecutionError(stdout=bytes('España', 'utf-8'), stderr=bytes('Zürich', 'utf-8'))
    executor = brick_executor.Executor(root_helper=None)
    exc = self.assertRaises(putils.ProcessExecutionError, executor._execute)
    self.assertEqual('España', exc.stdout)
    self.assertEqual('Zürich', exc.stderr)