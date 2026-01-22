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
def test_do_image_delete(self):
    args = argparse.Namespace(id=['image1', 'image2'])
    with mock.patch.object(self.gc.images, 'delete') as mocked_delete:
        mocked_delete.return_value = 0
        test_shell.do_image_delete(self.gc, args)
        self.assertEqual(2, mocked_delete.call_count)