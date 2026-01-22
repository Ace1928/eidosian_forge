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
@mock.patch.object(utils, 'exit')
@mock.patch.object(utils, 'print_err')
def test_do_image_delete_with_forbidden_ids(self, mocked_print_err, mocked_utils_exit):
    args = argparse.Namespace(id=['image1', 'image2'])
    with mock.patch.object(self.gc.images, 'delete') as mocked_delete:
        mocked_delete.side_effect = exc.HTTPForbidden
        test_shell.do_image_delete(self.gc, args)
        self.assertEqual(2, mocked_delete.call_count)
        self.assertEqual(2, mocked_print_err.call_count)
        mocked_utils_exit.assert_called_once_with()