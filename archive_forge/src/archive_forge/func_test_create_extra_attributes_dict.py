import argparse
from unittest import mock
import openstack
from osc_lib import exceptions
from openstackclient.network import common
from openstackclient.tests.unit import utils
def test_create_extra_attributes_dict(self):
    arglist = ['--known-attribute', 'known-value', '--extra-property', 'type=dict,name=extra_name,value=n1:v1;n2:v2']
    verifylist = [('known_attribute', 'known-value'), ('extra_properties', [{'name': 'extra_name', 'type': 'dict', 'value': 'n1:v1;n2:v2'}])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.network_client.test_create_action.assert_called_with(known_attribute='known-value', extra_name={'n1': 'v1', 'n2': 'v2'})