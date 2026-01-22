from unittest import mock
import urllib
import glance.common.exception as exception
from glance.common.scripts.image_import import main as image_import_script
from glance.common.scripts import utils
from glance.common import store_utils
import glance.tests.utils as test_utils
@mock.patch.object(image_import_script, 'create_image')
@mock.patch.object(image_import_script, 'set_image_data')
@mock.patch.object(store_utils, 'delete_image_location_from_backend')
def test_import_image_failed_with_expired_token(self, mock_delete_data, mock_set_img_data, mock_create_image):
    image_id = mock.ANY
    locations = ['location']
    image = mock.Mock(image_id=image_id, locations=locations)
    image_repo = mock.Mock()
    image_repo.get.side_effect = [image, exception.NotAuthenticated]
    image_factory = mock.ANY
    task_input = mock.Mock(image_properties=mock.ANY)
    uri = mock.ANY
    mock_create_image.return_value = image
    self.assertRaises(exception.NotAuthenticated, image_import_script.import_image, image_repo, image_factory, task_input, None, uri)
    self.assertEqual(1, mock_set_img_data.call_count)
    mock_delete_data.assert_called_once_with(mock_create_image().context, image_id, 'location')