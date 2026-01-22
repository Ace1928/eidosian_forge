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
def test_main_with_store_config_exception(self):
    with mock.patch.object(glance.common.config, 'parse_args') as mock_config:
        with mock.patch('sys.exit') as mock_exit:
            exc = store.exceptions.BadStoreConfiguration()
            mock_config.side_effect = exc
            glance.cmd.api.main()
            mock_exit.assert_called_once_with(3)