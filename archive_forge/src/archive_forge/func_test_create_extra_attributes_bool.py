import argparse
from unittest import mock
import openstack
from osc_lib import exceptions
from openstackclient.network import common
from openstackclient.tests.unit import utils
def test_create_extra_attributes_bool(self):
    arglist = ['--known-attribute', 'known-value', '--extra-property', 'type=bool,name=extra_name,value=TrUe']
    verifylist = [('known_attribute', 'known-value'), ('extra_properties', [{'name': 'extra_name', 'type': 'bool', 'value': 'TrUe'}])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.network_client.test_create_action.assert_called_with(known_attribute='known-value', extra_name=True)