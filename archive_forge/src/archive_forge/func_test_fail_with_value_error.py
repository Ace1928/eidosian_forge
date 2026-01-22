import io
import sys
from unittest import mock
import glance_store as store
from oslo_config import cfg
from oslo_log import log as logging
import glance.cmd.api
import glance.cmd.cache_cleaner
import glance.cmd.cache_pruner
import glance.common.config
from glance.common import exception as exc
import glance.common.wsgi
import glance.image_cache.cleaner
from glance.image_cache import prefetcher
import glance.image_cache.pruner
from glance.tests import utils as test_utils
def test_fail_with_value_error(self):
    with mock.patch('sys.stderr.write') as mock_stderr:
        with mock.patch('sys.exit') as mock_exit:
            exc_msg = 'A ValueError, LOL!'
            exc = ValueError(exc_msg)
            glance.cmd.api.fail(exc)
            mock_stderr.assert_called_once_with('ERROR: %s\n' % exc_msg)
            mock_exit.assert_called_once_with(4)