from unittest import mock
from oslo_log.fixture import logging_error as log_fixture
import testtools
import webob
import glance.api.common
from glance.common import exception
from glance.tests.unit import fixtures as glance_fixtures
@mock.patch('glance.async_.get_threadpool_model')
def test_get_thread_pool_log(self, mock_gtm):
    with mock.patch.object(glance.api.common, 'LOG') as mock_log:
        glance.api.common.get_thread_pool('test-pool')
        mock_log.debug.assert_called_once_with('Initializing named threadpool %r', 'test-pool')