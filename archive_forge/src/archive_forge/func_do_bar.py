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
@staticmethod
@cliutils.arg('--default-is-not-none', '--default_is_not_none', type=str, action='single_alias', help='Default value is not None and metavar not set.', default='bar')
def do_bar(cs, args):
    cliutils.print_dict({'key': args.default_is_not_none})