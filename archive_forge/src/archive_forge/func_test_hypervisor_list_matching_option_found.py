import json
from unittest import mock
from novaclient import exceptions as nova_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import hypervisor
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
def test_hypervisor_list_matching_option_found(self):
    arglist = ['--matching', self.hypervisors[0].name]
    verifylist = [('matching', self.hypervisors[0].name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.compute_sdk_client.hypervisors.return_value = [self.hypervisors[0]]
    self.data = ((self.hypervisors[0].id, self.hypervisors[0].name, self.hypervisors[1].hypervisor_type, self.hypervisors[1].host_ip, self.hypervisors[1].state),)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.hypervisors.assert_called_with(hypervisor_hostname_pattern=self.hypervisors[0].name, details=True)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, tuple(data))