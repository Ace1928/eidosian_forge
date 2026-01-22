from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor_profile
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
def test_create_without_driver_and_metainfo(self):
    arglist = ['--description', self.new_flavor_profile.description, '--project', self.new_flavor_profile.project_id, '--project-domain', self.domain.name, '--enable']
    verifylist = [('description', self.new_flavor_profile.description), ('project', self.new_flavor_profile.project_id), ('project_domain', self.domain.name), ('enable', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)