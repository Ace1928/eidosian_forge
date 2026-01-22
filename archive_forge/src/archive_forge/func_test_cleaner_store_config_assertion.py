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
@mock.patch('glance.async_.set_threadpool_model', new=mock.MagicMock())
def test_cleaner_store_config_assertion(self):
    failure = exc.GlanceException('This is what happens with http://')
    self.config(node_staging_uri='http://good.luck')
    self.mock_object(glance.common.wsgi.Server, 'start', self._raise(failure))
    exit = self.assertRaises(SystemExit, glance.cmd.api.main)
    self.assertEqual(99, exit.code)