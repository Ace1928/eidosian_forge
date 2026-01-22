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
@mock.patch('sys.stderr')
def test_image_create_missing_disk_format(self, __):
    e = self.assertRaises(exc.CommandError, self._run_command, '--os-image-api-version 2 image-create ' + '--file fake_src --container-format bare')
    self.assertEqual('error: Must provide --disk-format when using --file.', e.message)