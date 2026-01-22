from openstackclient.identity.v2_0 import endpoint
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_endpoint_list_long(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.endpoints_mock.list.assert_called_with()
    collist = ('ID', 'Region', 'Service Name', 'Service Type', 'PublicURL', 'AdminURL', 'InternalURL')
    self.assertEqual(collist, columns)
    datalist = ((self.fake_endpoint.id, self.fake_endpoint.region, self.fake_endpoint.service_name, self.fake_endpoint.service_type, self.fake_endpoint.publicurl, self.fake_endpoint.adminurl, self.fake_endpoint.internalurl),)
    self.assertEqual(datalist, tuple(data))