from openstackclient.identity.v3 import domain
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_domain_create_disable(self):
    arglist = ['--disable', self.domain.name]
    verifylist = [('disable', True), ('name', self.domain.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'name': self.domain.name, 'description': None, 'options': {}, 'enabled': False}
    self.domains_mock.create.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)