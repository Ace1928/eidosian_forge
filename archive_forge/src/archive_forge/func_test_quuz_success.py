import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from tempest.lib.cli import output_parser
from testtools import matchers
import manilaclient
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import shell
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
@ddt.data(('--list-like q=w', "['q=w']"), ('--list-like q=w --list_like q=w', "['q=w']"), ('--list-like q=w e=r t=y --list_like e=r t=y q=w', "['e=r', 'q=w', 't=y']"), ('--list_like q=w e=r t=y', "['e=r', 'q=w', 't=y']"))
@ddt.unpack
def test_quuz_success(self, options_str, expected_result):
    output = self.shell('quuz %s' % options_str)
    parsed_output = output_parser.details(output)
    self.assertEqual({'key': expected_result}, parsed_output)