from oslo_utils import reflection
import heat.api.openstack.v1 as api_v1
from heat.tests import common
def test_stack_data_template(self):
    self.assertRoute(self.m, '/aaaa/stacks/teststack/bbbb/template', 'GET', 'template', 'StackController', {'tenant_id': 'aaaa', 'stack_name': 'teststack', 'stack_id': 'bbbb'})
    self.assertRoute(self.m, '/aaaa/stacks/teststack/template', 'GET', 'lookup', 'StackController', {'tenant_id': 'aaaa', 'stack_name': 'teststack', 'path': 'template'})