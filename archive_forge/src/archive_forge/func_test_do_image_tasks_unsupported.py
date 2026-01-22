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
def test_do_image_tasks_unsupported(self):
    with mock.patch('glanceclient.common.utils.exit') as mock_exit:
        self._test_do_image_tasks(supported=False)
        mock_exit.assert_called_once_with('Server does not support image tasks API (v2.12)')