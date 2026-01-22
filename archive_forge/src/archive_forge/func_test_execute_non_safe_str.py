import threading
from unittest import mock
from oslo_concurrency import processutils as putils
from oslo_context import context as context_utils
from os_brick import executor as brick_executor
from os_brick.privileged import rootwrap
from os_brick.tests import base
@mock.patch('sys.stdin', encoding='UTF-8')
@mock.patch('os_brick.executor.priv_rootwrap.execute')
def test_execute_non_safe_str(self, execute_mock, stdin_mock):
    execute_mock.return_value = ('España', 'Zürich')
    executor = brick_executor.Executor(root_helper=None)
    stdout, stderr = executor._execute()
    self.assertEqual('España', stdout)
    self.assertEqual('Zürich', stderr)