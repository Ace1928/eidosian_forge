from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
@mock.patch.object(stacks_view, 'util', new=mock.Mock())
def test_doesnt_add_project_if_not_include_project(self):
    stack = {'stack_identity': {'stack_id': 'foo', 'tenant': 'bar'}}
    result = stacks_view.format_stack(self.request, stack, None, include_project=False)
    self.assertNotIn('project', result)