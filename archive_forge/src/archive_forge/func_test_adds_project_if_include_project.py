from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
@mock.patch.object(stacks_view, 'util', new=mock.Mock())
def test_adds_project_if_include_project(self):
    stack = {'stack_identity': {'stack_id': 'foo', 'tenant': 'bar'}}
    result = stacks_view.format_stack(self.request, stack, None, include_project=True)
    self.assertIn('project', result)
    self.assertEqual('bar', result['project'])