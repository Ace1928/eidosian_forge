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
def test_set_image_extra_properties(self, mock_log):
    mock_repo = mock.MagicMock()
    mock_image = mock_repo.get.return_value
    mock_image.image_id = IMAGE_ID1
    mock_image.extra_properties = {'os_glance_import_task': TASK_ID1}
    mock_image.status = 'bar'
    wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
    with wrapper as action:
        action.set_image_extra_properties({'os_glance_foo': 'bar'})
    self.assertEqual({'os_glance_import_task': TASK_ID1}, mock_image.extra_properties)
    mock_log.warning.assert_called()
    mock_log.warning.reset_mock()
    with wrapper as action:
        action.set_image_extra_properties({'os_glance_foo': 'bar', 'os_glance_baz': 'bat'})
    self.assertEqual({'os_glance_import_task': TASK_ID1}, mock_image.extra_properties)
    mock_log.warning.assert_called()
    mock_log.warning.reset_mock()
    with wrapper as action:
        action.set_image_extra_properties({'foo': 'bar', 'os_glance_foo': 'baz'})
    self.assertEqual({'foo': 'bar', 'os_glance_import_task': TASK_ID1}, mock_image.extra_properties)
    mock_log.warning.assert_called_once_with('Dropping %(key)s=%(val)s during metadata injection for %(image)s', {'key': 'os_glance_foo', 'val': 'baz', 'image': IMAGE_ID1})