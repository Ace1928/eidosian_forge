import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import registered_limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_registered_limit_set_resource_name(self):
    registered_limit = copy.deepcopy(identity_fakes.REGISTERED_LIMIT)
    resource_name = 'volumes'
    registered_limit['resource_name'] = resource_name
    self.registered_limit_mock.update.return_value = fakes.FakeResource(None, registered_limit, loaded=True)
    arglist = ['--resource-name', resource_name, identity_fakes.registered_limit_id]
    verifylist = [('resource_name', resource_name), ('registered_limit_id', identity_fakes.registered_limit_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.registered_limit_mock.update.assert_called_with(identity_fakes.registered_limit_id, service=None, resource_name=resource_name, default_limit=None, description=None, region=None)
    collist = ('default_limit', 'description', 'id', 'region_id', 'resource_name', 'service_id')
    self.assertEqual(collist, columns)
    datalist = (identity_fakes.registered_limit_default_limit, None, identity_fakes.registered_limit_id, None, resource_name, identity_fakes.service_id)
    self.assertEqual(datalist, data)