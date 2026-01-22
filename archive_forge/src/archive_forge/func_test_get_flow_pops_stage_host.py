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
def test_get_flow_pops_stage_host(self):
    import_flow.get_flow(task_id=TASK_ID1, task_type=TASK_TYPE, task_repo=self.mock_task_repo, image_repo=self.mock_image_repo, image_id=IMAGE_ID1, context=mock.MagicMock(), import_req=self.gd_task_input['import_req'])
    self.assertNotIn('os_glance_stage_host', self.mock_image.extra_properties)
    self.assertIn('os_glance_import_task', self.mock_image.extra_properties)