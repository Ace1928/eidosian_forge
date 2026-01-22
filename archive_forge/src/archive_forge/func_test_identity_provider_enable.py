import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_identity_provider_enable(self):
    """Enable Identity Provider.

        Set Identity Provider's ``enabled`` attribute to True.
        """

    def prepare(self):
        """Prepare fake return objects before the test is executed"""
        resources = fakes.FakeResource(None, copy.deepcopy(identity_fakes.IDENTITY_PROVIDER), loaded=True)
        self.identity_providers_mock.update.return_value = resources
    prepare(self)
    arglist = ['--enable', identity_fakes.idp_id, '--remote-id', identity_fakes.idp_remote_ids[0], '--remote-id', identity_fakes.idp_remote_ids[1]]
    verifylist = [('identity_provider', identity_fakes.idp_id), ('description', None), ('enable', True), ('disable', False), ('remote_id', identity_fakes.idp_remote_ids)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.identity_providers_mock.update.assert_called_with(identity_fakes.idp_id, enabled=True, remote_ids=identity_fakes.idp_remote_ids)