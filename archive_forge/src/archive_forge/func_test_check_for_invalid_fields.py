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
def test_check_for_invalid_fields(self):
    self.assertIsNone(utils.check_for_invalid_fields(['a', 'b'], ['a', 'b', 'c']))
    self.assertRaises(exc.CommandError, utils.check_for_invalid_fields, ['a', 'd'], ['a', 'b', 'c'])