import threading
from unittest import mock
from oslo_concurrency import processutils as putils
from oslo_context import context as context_utils
from os_brick import executor as brick_executor
from os_brick.privileged import rootwrap
from os_brick.tests import base
def test_none_execute(self):
    executor = brick_executor.Executor(root_helper=None, execute=None)
    self.assertEqual(rootwrap.execute, executor._Executor__execute)