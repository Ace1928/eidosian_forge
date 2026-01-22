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
def test_handle_json_or_file_arg_file(self):
    contents = '[{"step": "upgrade", "interface": "deploy"}]'
    with tempfile.NamedTemporaryFile(mode='w') as f:
        f.write(contents)
        f.flush()
        steps = utils.handle_json_or_file_arg(f.name)
    self.assertEqual(json.loads(contents), steps)