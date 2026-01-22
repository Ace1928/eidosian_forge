import argparse
from unittest import mock
import openstack
from osc_lib import exceptions
from openstackclient.network import common
from openstackclient.tests.unit import utils
def test_create_extra_attributes_string(self):
    arglist = ['--known-attribute', 'known-value', '--extra-property', 'type=str,name=extra_name,value=extra_value']
    verifylist = [('known_attribute', 'known-value'), ('extra_properties', [{'name': 'extra_name', 'type': 'str', 'value': 'extra_value'}])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.network_client.test_create_action.assert_called_with(known_attribute='known-value', extra_name='extra_value')