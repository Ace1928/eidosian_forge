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
def test_do_image_update_no_user_props(self):
    args = self._make_args({'id': 'pass', 'name': 'IMG-01', 'disk_format': 'vhd', 'container_format': 'bare'})
    with mock.patch.object(self.gc.images, 'update') as mocked_update:
        ignore_fields = ['self', 'access', 'file', 'schema']
        expect_image = dict([(field, field) for field in ignore_fields])
        expect_image['id'] = 'pass'
        expect_image['name'] = 'IMG-01'
        expect_image['disk_format'] = 'vhd'
        expect_image['container_format'] = 'bare'
        mocked_update.return_value = expect_image
        test_shell.do_image_update(self.gc, args)
        mocked_update.assert_called_once_with('pass', None, name='IMG-01', disk_format='vhd', container_format='bare')
        utils.print_dict.assert_called_once_with({'id': 'pass', 'name': 'IMG-01', 'disk_format': 'vhd', 'container_format': 'bare'})