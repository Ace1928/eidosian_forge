from unittest import mock
from zunclient.tests.unit.v1 import shell_test_base
@mock.patch('zunclient.v1.images.ImageManager.list')
def test_zun_image_list_failure(self, mock_list):
    self._test_arg_failure('image-list --wrong', self._unrecognized_arg_error)
    self.assertFalse(mock_list.called)