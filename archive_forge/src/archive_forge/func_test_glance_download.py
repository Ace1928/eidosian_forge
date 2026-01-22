from unittest import mock
import urllib.error
from glance_store._drivers import filesystem
from oslo_config import cfg
from oslo_utils.fixture import uuidsentinel
from glance.async_.flows._internal_plugins import glance_download
from glance.async_.flows import api_image_import
import glance.common.exception
import glance.context
from glance import domain
import glance.tests.utils as test_utils
@mock.patch.object(filesystem.Store, 'add')
@mock.patch('glance.async_.utils.get_glance_endpoint')
def test_glance_download(self, mock_gge, mock_add):
    mock_gge.return_value = 'https://other.cloud.foo/image'
    glance_download_task = glance_download._DownloadGlanceImage(self.context, self.task.task_id, self.task_type, self.action_wrapper, ['foo'], 'RegionTwo', uuidsentinel.remote_image, 'public')
    with mock.patch('urllib.request') as mock_request:
        mock_add.return_value = ['path', 12345]
        self.assertEqual(glance_download_task.execute(12345), 'path')
        mock_add.assert_called_once_with(self.image_id, mock_request.urlopen.return_value, 0)
        mock_request.Request.assert_called_once_with('https://other.cloud.foo/image/v2/images/%s/file' % uuidsentinel.remote_image, headers={'X-Auth-Token': self.context.auth_token})
    mock_gge.assert_called_once_with(self.context, 'RegionTwo', 'public')