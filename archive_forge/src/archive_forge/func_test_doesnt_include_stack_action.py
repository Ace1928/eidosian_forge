from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
def test_doesnt_include_stack_action(self):
    stack = {'stack_action': 'CREATE'}
    result = stacks_view.format_stack(self.request, stack)
    self.assertEqual({}, result)