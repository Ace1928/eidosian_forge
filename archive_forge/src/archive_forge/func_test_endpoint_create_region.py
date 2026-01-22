from openstackclient.identity.v3 import endpoint
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_endpoint_create_region(self):
    arglist = [self.service.id, self.endpoint.interface, self.endpoint.url, '--region', self.endpoint.region]
    verifylist = [('enabled', True), ('service', self.service.id), ('interface', self.endpoint.interface), ('url', self.endpoint.url), ('region', self.endpoint.region)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'service': self.service.id, 'url': self.endpoint.url, 'interface': self.endpoint.interface, 'enabled': True, 'region': self.endpoint.region}
    self.endpoints_mock.create.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)
    datalist = (True, self.endpoint.id, self.endpoint.interface, self.endpoint.region, self.service.id, self.service.name, self.service.type, self.endpoint.url)
    self.assertEqual(datalist, data)