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
@mock.patch('sys.stderr')
def test_image_list_invalid_min_size_parameter(self, __):
    self.assertRaises(SystemExit, self.run_command, 'image-list --size-min 10gb')