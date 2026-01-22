from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import local_ip_association
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
def test_local_ip_association_list_all_options(self):
    arglist = ['--fixed-port', self.local_ip_associations[0].fixed_port_id, '--fixed-ip', self.local_ip_associations[0].fixed_ip, '--host', self.local_ip_associations[0].host, self.local_ip_associations[0].local_ip_id]
    verifylist = [('fixed_port', self.local_ip_associations[0].fixed_port_id), ('fixed_ip', self.local_ip_associations[0].fixed_ip), ('host', self.local_ip_associations[0].host), ('local_ip', self.local_ip_associations[0].local_ip_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    attrs = {'fixed_port_id': self.local_ip_associations[0].fixed_port_id, 'fixed_ip': self.local_ip_associations[0].fixed_ip, 'host': self.local_ip_associations[0].host}
    self.network_client.local_ip_associations.assert_called_once_with(self.local_ip, **attrs)
    self.assertEqual(set(self.columns), set(columns))
    self.assertEqual(set(self.data), set(list(data)))