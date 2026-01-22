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
@mock.patch('glance.async_.set_threadpool_model')
@mock.patch.object(prefetcher, 'Prefetcher')
def test_supported_default_store(self, mock_prefetcher, mock_set_model):
    self.config(group='glance_store', default_store='file')
    glance.cmd.api.main()
    mock_set_model.assert_called_once_with('eventlet')