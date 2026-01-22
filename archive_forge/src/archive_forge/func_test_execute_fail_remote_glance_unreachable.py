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
@mock.patch('urllib.request')
@mock.patch('glance.async_.utils.get_glance_endpoint')
def test_execute_fail_remote_glance_unreachable(self, mock_gge, mock_r):
    action = self.wrapper.__enter__.return_value
    mock_r.urlopen.side_effect = urllib.error.HTTPError('/file', 400, 'Test Fail', {}, None)
    task = import_flow._ImportMetadata(TASK_ID1, TASK_TYPE, self.context, self.wrapper, self.import_req)
    self.assertRaises(urllib.error.HTTPError, task.execute)
    action.assert_not_called()