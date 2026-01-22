from keystoneclient import exceptions as identity_exc
from osc_lib import exceptions
from openstackclient.identity.v2_0 import service
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_service_list_no_options(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.services_mock.list.assert_called_with()
    collist = ('ID', 'Name', 'Type')
    self.assertEqual(collist, columns)
    datalist = ((self.fake_service.id, self.fake_service.name, self.fake_service.type),)
    self.assertEqual(datalist, tuple(data))