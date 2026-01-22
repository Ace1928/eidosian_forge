from unittest import mock
from zunclient.tests.unit.v1 import shell_test_base
@mock.patch('zunclient.v1.images.ImageManager.get')
def test_zun_image_show_success(self, mock_get):
    self._test_arg_success('image-show 111')
    self.assertTrue(mock_get.called)