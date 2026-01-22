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
def test_do_image_deactivate(self):
    args = argparse.Namespace(id='image1')
    with mock.patch.object(self.gc.images, 'deactivate') as mocked_deactivate:
        mocked_deactivate.return_value = 0
        test_shell.do_image_deactivate(self.gc, args)
        self.assertEqual(1, mocked_deactivate.call_count)