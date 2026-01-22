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
@mock.patch('glance.async_.flows.api_image_import.image_import')
@mock.patch('glance_store.get_store_from_store_identifier')
def test_init_copy_flow_as_non_owner(self, mock_gs, mock_import):
    img_repo = mock.MagicMock()
    admin_repo = mock.MagicMock()
    fake_req = {'method': {'name': 'copy-image'}, 'backend': ['cheap']}
    fake_img = mock.MagicMock()
    fake_img.id = IMAGE_ID1
    fake_img.status = 'active'
    fake_img.extra_properties = {'os_glance_import_task': TASK_ID1}
    admin_repo.get.return_value = fake_img
    import_flow.get_flow(task_id=TASK_ID1, task_type=TASK_TYPE, task_repo=mock.MagicMock(), image_repo=img_repo, admin_repo=admin_repo, image_id=IMAGE_ID1, import_req=fake_req, context=self.context, backend=['cheap'])
    admin_repo.save.assert_called_once_with(fake_img, 'active')
    img_repo.save.assert_not_called()