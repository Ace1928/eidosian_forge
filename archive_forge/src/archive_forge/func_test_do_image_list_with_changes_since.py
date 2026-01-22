import argparse
import io
import json
import os
from unittest import mock
import subprocess
import tempfile
import testtools
from glanceclient import exc
from glanceclient import shell
import glanceclient.v1.client as client
import glanceclient.v1.images
import glanceclient.v1.shell as v1shell
from glanceclient.tests import utils
def test_do_image_list_with_changes_since(self):
    input = {'name': None, 'limit': None, 'status': None, 'container_format': 'bare', 'size_min': None, 'size_max': None, 'is_public': True, 'disk_format': 'raw', 'page_size': 20, 'visibility': True, 'member_status': 'Fake', 'owner': 'test', 'checksum': 'fake_checksum', 'tag': 'fake tag', 'properties': [], 'sort_key': None, 'sort_dir': None, 'all_tenants': False, 'human_readable': True, 'changes_since': '2011-1-1'}
    args = self._make_args(input)
    with mock.patch.object(self.gc.images, 'list') as mocked_list:
        mocked_list.return_value = {}
        v1shell.do_image_list(self.gc, args)
        exp_img_filters = {'container_format': 'bare', 'changes-since': '2011-1-1', 'disk_format': 'raw', 'is_public': True}
        mocked_list.assert_called_once_with(sort_dir=None, sort_key=None, owner='test', page_size=20, filters=exp_img_filters)