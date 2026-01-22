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
def test_check_task_lock(self, mock_log):
    mock_repo = mock.MagicMock()
    wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
    image = mock.MagicMock(image_id=IMAGE_ID1)
    image.extra_properties = {'os_glance_import_task': TASK_ID1}
    mock_repo.get.return_value = image
    self._grab_image(wrapper)
    mock_log.error.assert_not_called()
    image.extra_properties['os_glance_import_task'] = 'somethingelse'
    self.assertRaises(exception.TaskAbortedError, self._grab_image, wrapper)
    mock_log.error.assert_called_once_with('Image %(image)s import task %(task)s attempted to take action on image, but other task %(other)s holds the lock; Aborting.', {'image': image.image_id, 'task': TASK_ID1, 'other': 'somethingelse'})