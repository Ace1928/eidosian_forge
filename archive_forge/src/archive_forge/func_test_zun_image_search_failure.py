from unittest import mock
from zunclient.tests.unit.v1 import shell_test_base
@mock.patch('zunclient.v1.images.ImageManager.search_image')
def test_zun_image_search_failure(self, mock_search_image):
    self._test_arg_failure('image-search --wrong 1111', self._unrecognized_arg_error)
    self.assertFalse(mock_search_image.called)