import threading
from unittest import mock
from oslo_concurrency import processutils as putils
from oslo_context import context as context_utils
from os_brick import executor as brick_executor
from os_brick.privileged import rootwrap
from os_brick.tests import base
def test_fake_execute(self):
    mock_execute = mock.Mock()
    executor = brick_executor.Executor(root_helper=None, execute=mock_execute)
    self.assertEqual(mock_execute, executor._Executor__execute)