import copy
from openstackclient.identity.v3 import region
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_region_create_description(self):
    arglist = [identity_fakes.region_id, '--description', identity_fakes.region_description]
    verifylist = [('region', identity_fakes.region_id), ('description', identity_fakes.region_description)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'description': identity_fakes.region_description, 'id': identity_fakes.region_id, 'parent_region': None}
    self.regions_mock.create.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)