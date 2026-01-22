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
@mock.patch.object(os, 'stat')
@mock.patch.object(os, 'remove')
def test_image_convert_success(self, mock_os_remove, mock_os_stat):
    mock_os_remove.return_value = None
    mock_os_stat.return_value.st_size = 123
    image_convert = image_conversion._ConvertImage(self.context, self.task.task_id, self.task_type, self.wrapper)
    self.task_repo.get.return_value = self.task
    image = mock.MagicMock(image_id=self.image_id, virtual_size=None, extra_properties={'os_glance_import_task': self.task.task_id}, disk_format='qcow2')
    self.img_repo.get.return_value = image
    with mock.patch.object(processutils, 'execute') as exc_mock:
        exc_mock.return_value = ('', None)
        with mock.patch.object(json, 'loads') as jloads_mock:
            jloads_mock.return_value = {'format': 'raw', 'virtual-size': 456}
            image_convert.execute('file:///test/path.raw')
            self.assertIn('-f', exc_mock.call_args[0])
            self.assertEqual('qcow2', image.disk_format)
    self.assertEqual('bare', image.container_format)
    self.assertEqual('qcow2', image.disk_format)
    self.assertEqual(456, image.virtual_size)
    self.assertEqual(123, image.size)