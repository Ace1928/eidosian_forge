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
def test_image_convert_same_format_does_nothing(self):
    convert = self._setup_image_convert_info_fail()
    with mock.patch.object(processutils, 'execute') as exc_mock:
        exc_mock.return_value = ('{"format": "qcow2", "virtual-size": 123}', '')
        convert.execute('file:///test/path.qcow')
        exc_mock.assert_called_once_with('qemu-img', 'info', '--output=json', '/test/path.qcow', prlimit=async_utils.QEMU_IMG_PROC_LIMITS, python_exec=convert.python, log_errors=processutils.LOG_ALL_ERRORS)
    image = self.img_repo.get.return_value
    self.assertEqual(123, image.virtual_size)