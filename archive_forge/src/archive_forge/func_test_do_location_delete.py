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
def test_do_location_delete(self):
    gc = self.gc
    loc_set = set(['http://foo/bar', 'http://spam/ham'])
    args = self._make_args({'id': 'pass', 'url': loc_set})
    with mock.patch.object(gc.images, 'delete_locations') as mocked_rmloc:
        test_shell.do_location_delete(self.gc, args)
        mocked_rmloc.assert_called_once_with('pass', loc_set)