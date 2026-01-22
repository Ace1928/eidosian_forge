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
def test_raises_when_image_deleted(self):
    img_repo = mock.MagicMock()
    task_repo = mock.MagicMock()
    wrapper = import_flow.ImportActionWrapper(img_repo, IMAGE_ID1, TASK_ID1)
    image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, wrapper, 'http://url', 'store1', False, True)
    image = self.img_factory.new_image(image_id=UUID1)
    image.status = 'deleted'
    img_repo.get.return_value = image
    self.assertRaises(exception.ImportTaskError, image_import.execute)