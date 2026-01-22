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
def test_do_location_add(self):
    gc = self.gc
    loc = {'url': 'http://foo.com/', 'metadata': {'foo': 'bar'}, 'validation_data': {'checksum': 'csum', 'os_hash_algo': 'algo', 'os_hash_value': 'value'}}
    args = {'id': 'pass', 'url': loc['url'], 'metadata': json.dumps(loc['metadata']), 'checksum': 'csum', 'hash_algo': 'algo', 'hash_value': 'value'}
    with mock.patch.object(gc.images, 'add_location') as mocked_addloc:
        expect_image = {'id': 'pass', 'locations': [loc]}
        mocked_addloc.return_value = expect_image
        test_shell.do_location_add(self.gc, self._make_args(args))
        mocked_addloc.assert_called_once_with('pass', loc['url'], loc['metadata'], validation_data=loc['validation_data'])
        utils.print_dict.assert_called_once_with(expect_image)