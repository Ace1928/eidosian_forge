from unittest import mock
from glance_store._drivers import filesystem
from oslo_config import cfg
from glance.async_.flows._internal_plugins import web_download
from glance.async_.flows import api_image_import
import glance.common.exception
import glance.common.scripts.utils as script_utils
from glance import domain
import glance.tests.utils as test_utils
@mock.patch.object(filesystem.Store, 'add')
def test_web_download_fails_when_data_size_different(self, mock_add):
    with mock.patch.object(script_utils, 'get_image_data_iter') as mock_iter:
        mock_iter.return_value.headers = {'content-length': '4'}
        mock_add.return_value = ['path', 3]
        self.assertRaises(glance.common.exception.ImportTaskError, self.web_download_task.execute)