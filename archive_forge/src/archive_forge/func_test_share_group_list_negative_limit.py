import argparse
from unittest import mock
import uuid
from osc_lib import exceptions
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient.osc import utils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_groups as osc_share_groups
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_group_list_negative_limit(self):
    arglist = ['--limit', '-2']
    verifylist = [('limit', -2)]
    self.assertRaises(argparse.ArgumentTypeError, self.check_parser, self.cmd, arglist, verifylist)