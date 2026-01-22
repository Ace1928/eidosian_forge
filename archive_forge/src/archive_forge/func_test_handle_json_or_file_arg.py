import builtins
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.tests.unit import utils as test_utils
def test_handle_json_or_file_arg(self):
    cleansteps = '[{"step": "upgrade", "interface": "deploy"}]'
    steps = utils.handle_json_or_file_arg(cleansteps)
    self.assertEqual(json.loads(cleansteps), steps)