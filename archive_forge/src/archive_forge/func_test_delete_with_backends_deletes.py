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
def test_delete_with_backends_deletes(self):
    task = import_flow._DeleteFromFS(TASK_ID1, TASK_TYPE)
    self.config(enabled_backends='file:foo')
    with mock.patch.object(import_flow.store_api, 'delete') as mock_del:
        task.execute(mock.sentinel.path)
        mock_del.assert_called_once_with(mock.sentinel.path, 'os_glance_staging_store')