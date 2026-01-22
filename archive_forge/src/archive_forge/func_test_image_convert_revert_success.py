import json
import os
from unittest import mock
import glance_store
from oslo_concurrency import processutils
from oslo_config import cfg
import glance.async_.flows.api_image_import as import_flow
import glance.async_.flows.plugins.image_conversion as image_conversion
from glance.async_ import utils as async_utils
from glance.common import utils
from glance import domain
from glance import gateway
import glance.tests.utils as test_utils
@mock.patch.object(os, 'remove')
def test_image_convert_revert_success(self, mock_os_remove):
    mock_os_remove.return_value = None
    image_convert = image_conversion._ConvertImage(self.context, self.task.task_id, self.task_type, self.wrapper)
    self.task_repo.get.return_value = self.task
    with mock.patch.object(processutils, 'execute') as exc_mock:
        exc_mock.return_value = ('', None)
        with mock.patch.object(os.path, 'exists') as os_exists_mock:
            os_exists_mock.return_value = True
            image_convert.revert(result=mock.MagicMock())
            self.assertEqual(1, mock_os_remove.call_count)