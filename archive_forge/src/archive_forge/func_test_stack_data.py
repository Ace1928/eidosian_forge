from oslo_utils import reflection
import heat.api.openstack.v1 as api_v1
from heat.tests import common
def test_stack_data(self):
    self.assertRoute(self.m, '/aaaa/stacks/teststack', 'GET', 'lookup', 'StackController', {'tenant_id': 'aaaa', 'stack_name': 'teststack'})
    self.assertRoute(self.m, '/aaaa/stacks/arn:openstack:heat::6548ab64fbda49deb188851a3b7d8c8b:stacks/stack-1411-06/1c5d9bb2-3464-45e2-a728-26dfa4e1d34a', 'GET', 'lookup', 'StackController', {'tenant_id': 'aaaa', 'stack_name': 'arn:openstack:heat::6548ab64fbda49deb188851a3b7d8c8b:stacks/stack-1411-06/1c5d9bb2-3464-45e2-a728-26dfa4e1d34a'})
    self.assertRoute(self.m, '/aaaa/stacks/teststack/resources', 'GET', 'lookup', 'StackController', {'tenant_id': 'aaaa', 'stack_name': 'teststack', 'path': 'resources'})
    self.assertRoute(self.m, '/aaaa/stacks/teststack/events', 'GET', 'lookup', 'StackController', {'tenant_id': 'aaaa', 'stack_name': 'teststack', 'path': 'events'})
    self.assertRoute(self.m, '/aaaa/stacks/teststack/bbbb', 'GET', 'show', 'StackController', {'tenant_id': 'aaaa', 'stack_name': 'teststack', 'stack_id': 'bbbb'})