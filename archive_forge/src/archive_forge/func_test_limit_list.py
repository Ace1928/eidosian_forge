import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_limit_list(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.limit_mock.list.assert_called_with(service=None, resource_name=None, region=None, project=None)
    collist = ('ID', 'Project ID', 'Service ID', 'Resource Name', 'Resource Limit', 'Description', 'Region ID')
    self.assertEqual(collist, columns)
    datalist = ((identity_fakes.limit_id, identity_fakes.project_id, identity_fakes.service_id, identity_fakes.limit_resource_name, identity_fakes.limit_resource_limit, None, None),)
    self.assertEqual(datalist, tuple(data))