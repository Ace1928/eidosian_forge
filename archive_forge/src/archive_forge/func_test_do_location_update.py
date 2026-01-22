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
def test_do_location_update(self):
    gc = self.gc
    loc = {'url': 'http://foo.com/', 'metadata': {'foo': 'bar'}}
    args = self._make_args({'id': 'pass', 'url': loc['url'], 'metadata': json.dumps(loc['metadata'])})
    with mock.patch.object(gc.images, 'update_location') as mocked_modloc:
        expect_image = {'id': 'pass', 'locations': [loc]}
        mocked_modloc.return_value = expect_image
        test_shell.do_location_update(self.gc, args)
        mocked_modloc.assert_called_once_with('pass', loc['url'], loc['metadata'])
        utils.print_dict.assert_called_once_with(expect_image)