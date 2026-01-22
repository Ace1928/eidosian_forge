import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
def test_write_to_completion_cache(self):
    manager = base.Manager()
    manager.write_to_completion_cache('non-exist', 'val')
    manager._mock_cache = mock.Mock()
    manager._mock_cache.write = mock.Mock(return_value=None)
    manager.write_to_completion_cache('mock', 'val')
    self.assertEqual(1, manager._mock_cache.write.call_count)