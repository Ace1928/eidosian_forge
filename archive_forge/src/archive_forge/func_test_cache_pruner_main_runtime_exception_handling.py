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
@mock.patch.object(glance.image_cache.base.CacheApp, '__init__')
def test_cache_pruner_main_runtime_exception_handling(self, mock_cache):
    mock_cache.return_value = None
    self.mock_object(glance.image_cache.pruner.Pruner, 'run', self._raise(RuntimeError))
    exit = self.assertRaises(SystemExit, glance.cmd.cache_pruner.main)
    self.assertEqual('ERROR: ', exit.code)