import argparse
import copy
import itertools
from unittest import mock
from osc_lib import exceptions
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import load_balancer
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_load_balancer_list_with_name(self):
    arglist = ['--name', 'rainbarrel']
    verifylist = [('name', 'rainbarrel')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.api_mock.load_balancer_list.assert_called_with(name='rainbarrel')
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))