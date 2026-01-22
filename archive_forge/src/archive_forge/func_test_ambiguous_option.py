import io
import re
import sys
from unittest import mock
import fixtures
from keystoneauth1 import fixture
from testtools import matchers
from zunclient import api_versions
from zunclient import exceptions
import zunclient.shell
from zunclient.tests.unit import utils
def test_ambiguous_option(self):
    self.parser.add_argument('--tic')
    self.parser.add_argument('--tac')
    try:
        self.parser.parse_args(['--t'])
    except SystemExit as err:
        self.assertEqual(2, err.code)
    else:
        self.fail('SystemExit not raised')