from oslo_utils import reflection
import heat.api.openstack.v1 as api_v1
from heat.tests import common
def test_stack_collection(self):
    self.assertRoute(self.m, '/aaaa/stacks', 'GET', 'index', 'StackController', {'tenant_id': 'aaaa'})
    self.assertRoute(self.m, '/aaaa/stacks', 'POST', 'create', 'StackController', {'tenant_id': 'aaaa'})
    self.assertRoute(self.m, '/aaaa/stacks/preview', 'POST', 'preview', 'StackController', {'tenant_id': 'aaaa'})
    self.assertRoute(self.m, '/aaaa/stacks/detail', 'GET', 'detail', 'StackController', {'tenant_id': 'aaaa'})