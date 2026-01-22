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
def test_sort_key_and_sort_dir(self):
    self.args.sort_key = 'field'
    self.args.sort_dir = 'desc'
    self.expected_params.update({'sort_key': 'field', 'sort_dir': 'desc'})
    self.assertEqual(self.expected_params, utils.common_params_for_list(self.args, ['field'], []))