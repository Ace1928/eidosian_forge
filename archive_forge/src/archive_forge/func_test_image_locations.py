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
def test_image_locations(self):
    mock_repo = mock.MagicMock()
    mock_image = mock_repo.get.return_value
    mock_image.image_id = IMAGE_ID1
    mock_image.extra_properties = {'os_glance_import_task': TASK_ID1}
    mock_image.locations = {'some': {'complex': ['structure']}}
    wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
    with wrapper as action:
        self.assertEqual({'some': {'complex': ['structure']}}, action.image_locations)
        action.image_locations['foo'] = 'bar'
    self.assertEqual({'some': {'complex': ['structure']}}, mock_image.locations)