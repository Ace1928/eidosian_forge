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
def test_handle_json_or_file_arg_bad_json(self):
    cleansteps = '{foo invalid: json{'
    self.assertRaisesRegex(exc.InvalidAttribute, 'is not a file and cannot be parsed as JSON', utils.handle_json_or_file_arg, cleansteps)