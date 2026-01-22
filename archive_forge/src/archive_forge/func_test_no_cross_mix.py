import threading
from unittest import mock
from oslo_concurrency import processutils as putils
from oslo_context import context as context_utils
from os_brick import executor as brick_executor
from os_brick.privileged import rootwrap
from os_brick.tests import base
def test_no_cross_mix(self):
    """Test there's no shared global context between threads."""
    result = []
    contexts = [[], [], []]
    threads = [threading.Thread(target=self._run_test, args=[self.test_with_context, contexts[0], result]), threading.Thread(target=self._run_test, args=[self.test_no_context, contexts[1], result]), threading.Thread(target=self._run_test, args=[self.test_with_context, contexts[2], result])]
    self._run_threads(threads)
    self.assertEqual([True, True, True], result)
    self.assertNotEqual(contexts[0], contexts[2])