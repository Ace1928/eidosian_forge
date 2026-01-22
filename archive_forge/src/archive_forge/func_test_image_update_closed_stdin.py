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
def test_image_update_closed_stdin(self):
    """Test image update with a closed stdin.

        Supply glanceclient with a closed stdin, and perform an image
        update to an active image. Glanceclient should not attempt to read
        stdin.
        """
    os.close(0)
    self._do_update()
    self.assertTrue('data' not in self.collected_args[1] or self.collected_args[1]['data'] is None)