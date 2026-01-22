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
def test_verify_active_status(self):
    fake_img = mock.MagicMock(status='active', extra_properties={'os_glance_import_task': TASK_ID1})
    mock_repo = mock.MagicMock()
    mock_repo.get.return_value = fake_img
    wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
    task = import_flow._VerifyImageState(TASK_ID1, TASK_TYPE, wrapper, 'anything!')
    task.execute()
    fake_img.status = 'importing'
    self.assertRaises(import_flow._NoStoresSucceeded, task.execute)