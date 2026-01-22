import copy
from openstackclient.identity.v3 import service_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as service_fakes
def test_service_provider_enable(self):
    """Enable Service Provider.

        Set Service Provider's ``enabled`` attribute to True.
        """

    def prepare(self):
        """Prepare fake return objects before the test is executed"""
        resources = fakes.FakeResource(None, copy.deepcopy(service_fakes.SERVICE_PROVIDER), loaded=True)
        self.service_providers_mock.update.return_value = resources
    prepare(self)
    arglist = ['--enable', service_fakes.sp_id]
    verifylist = [('service_provider', service_fakes.sp_id), ('enable', True), ('disable', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.service_providers_mock.update.assert_called_with(service_fakes.sp_id, enabled=True, description=None, auth_url=None, sp_url=None)