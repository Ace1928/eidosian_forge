import argparse
from copy import deepcopy
import io
import json
import os
from unittest import mock
import sys
import tempfile
import testtools
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell
from glanceclient.v2 import shell as test_shell  # noqa
@mock.patch('glanceclient.common.utils.exit')
def test_neg_image_import_copy_image_not_active(self, mock_utils_exit):
    expected_msg = "The 'copy-image' import method can only be used on an image with status 'active'."
    mock_utils_exit.side_effect = self._mock_utils_exit
    args = self._make_args({'id': 'IMG-02', 'uri': None, 'import_method': 'copy-image', 'disk_format': 'raw', 'container_format': 'bare', 'from_create': False, 'stores': 'file1,file2'})
    with mock.patch.object(self.gc.images, 'get_stores_info') as mocked_stores_info:
        with mock.patch.object(self.gc.images, 'get') as mocked_get:
            with mock.patch.object(self.gc.images, 'get_import_info') as mocked_info:
                mocked_stores_info.return_value = self.stores_info_response
                mocked_get.return_value = {'status': 'uploading', 'container_format': 'bare', 'disk_format': 'raw'}
                mocked_info.return_value = self.import_info_response
                try:
                    test_shell.do_image_import(self.gc, args)
                    self.fail('utils.exit should have been called')
                except SystemExit:
                    pass
        mock_utils_exit.assert_called_once_with(expected_msg)