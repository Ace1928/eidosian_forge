from openstackclient.identity.v3 import endpoint
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_endpoint_set_region(self):
    arglist = ['--region', 'e-rzzz', self.endpoint.id]
    verifylist = [('region', 'e-rzzz'), ('endpoint', self.endpoint.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'enabled': None, 'interface': None, 'url': None, 'region': 'e-rzzz', 'service': None}
    self.endpoints_mock.update.assert_called_with(self.endpoint.id, **kwargs)
    self.assertIsNone(result)