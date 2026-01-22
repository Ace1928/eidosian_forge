import threading
from unittest import mock
from oslo_concurrency import processutils as putils
from oslo_context import context as context_utils
from os_brick import executor as brick_executor
from os_brick.privileged import rootwrap
from os_brick.tests import base
def test_normal_thread(self):
    """Test normal threads don't inherit parent's context."""
    context = context_utils.RequestContext()
    context.update_store()
    self._do_test(threading.Thread, None)