from oslo_utils import uuidutils
import testtools
from webob import exc
from neutron_lib.api import attributes
from neutron_lib.api import converters
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib import constants
from neutron_lib import context
from neutron_lib import exceptions
from neutron_lib.tests import _base as base
def test__validate_privileges_service_role_other_tenant(self):
    project_id = 'fake_project'
    ctx = context.Context(project_id='fake_project2', roles=['service'])
    res_dict = {'project_id': project_id}
    try:
        attributes._validate_privileges(ctx, res_dict)
    except exc.HTTPBadRequest:
        self.fail('HTTPBadRequest exception should not be raised.')