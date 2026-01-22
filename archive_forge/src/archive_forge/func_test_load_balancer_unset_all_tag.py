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
def test_load_balancer_unset_all_tag(self):
    self.api_mock.load_balancer_show.return_value = {'tags': ['foo', 'bar']}
    arglist = [self._lb.id, '--all-tag']
    verifylist = [('loadbalancer', self._lb.id), ('all_tag', True)]
    try:
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
    except Exception as e:
        self.fail('%s raised unexpectedly' % e)
    self.api_mock.load_balancer_set.assert_called_once_with(self._lb.id, json={'loadbalancer': {'tags': []}})