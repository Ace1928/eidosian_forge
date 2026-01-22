import argparse
from unittest import mock
import openstack
from osc_lib import exceptions
from openstackclient.network import common
from openstackclient.tests.unit import utils
def test_take_action_compute(self):
    arglist = ['common', 'compute']
    verifylist = [('common', 'common'), ('compute', 'compute')]
    self.app.client_manager.network_endpoint_enabled = False
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.compute_client.compute_action.assert_called_with(parsed_args)
    self.assertEqual('take_action_compute', result)