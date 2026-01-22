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
def test_marker_and_limit(self):
    self.args.marker = 'foo'
    self.args.limit = 42
    self.expected_params.update({'marker': 'foo', 'limit': 42})
    self.assertEqual(self.expected_params, utils.common_params_for_list(self.args, [], []))