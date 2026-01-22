import sys
from unittest import mock
import urllib.error
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_utils import units
import taskflow
import glance.async_.flows.api_image_import as import_flow
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance import context
from glance.domain import ExtraProperties
from glance import gateway
import glance.tests.utils as test_utils
from cursive import exception as cursive_exception
@mock.patch.object(import_flow, 'LOG')
def test_merge_store_logs_info(self, mock_log):
    self.actions.merge_store_list('stores', ['foo,bar'], subtract=True)
    mock_log.debug.assert_has_calls([mock.call('Stores %(stores)s not in %(key)s for image %(image_id)s', {'image_id': IMAGE_ID1, 'key': 'stores', 'stores': 'foo,bar'}), mock.call('Image %(image_id)s %(key)s=%(stores)s', {'image_id': IMAGE_ID1, 'key': 'stores', 'stores': ''})])
    mock_log.debug.reset_mock()
    self.actions.merge_store_list('stores', ['foo'])
    self.assertEqual('foo', self.image.extra_properties['stores'])
    mock_log.debug.reset_mock()
    self.actions.merge_store_list('stores', ['bar'], subtract=True)
    self.assertEqual('foo', self.image.extra_properties['stores'])
    mock_log.debug.assert_has_calls([mock.call('Stores %(stores)s not in %(key)s for image %(image_id)s', {'image_id': IMAGE_ID1, 'key': 'stores', 'stores': 'bar'}), mock.call('Image %(image_id)s %(key)s=%(stores)s', {'image_id': IMAGE_ID1, 'key': 'stores', 'stores': 'foo'})])