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
def test_do_image_reactivate(self):
    args = argparse.Namespace(id='image1')
    with mock.patch.object(self.gc.images, 'reactivate') as mocked_reactivate:
        mocked_reactivate.return_value = 0
        test_shell.do_image_reactivate(self.gc, args)
        self.assertEqual(1, mocked_reactivate.call_count)