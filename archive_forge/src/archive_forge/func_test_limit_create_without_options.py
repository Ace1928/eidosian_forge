import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_limit_create_without_options(self):
    self.limit_mock.create.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.LIMIT), loaded=True)
    resource_limit = 15
    arglist = ['--project', identity_fakes.project_id, '--service', identity_fakes.service_id, '--resource-limit', str(resource_limit), identity_fakes.limit_resource_name]
    verifylist = [('project', identity_fakes.project_id), ('service', identity_fakes.service_id), ('resource_name', identity_fakes.limit_resource_name), ('resource_limit', resource_limit)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'description': None, 'region': None}
    self.limit_mock.create.assert_called_with(self.project, self.service, identity_fakes.limit_resource_name, resource_limit, **kwargs)
    collist = ('description', 'id', 'project_id', 'region_id', 'resource_limit', 'resource_name', 'service_id')
    self.assertEqual(collist, columns)
    datalist = (None, identity_fakes.limit_id, identity_fakes.project_id, None, resource_limit, identity_fakes.limit_resource_name, identity_fakes.service_id)
    self.assertEqual(datalist, data)