from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
def test_filter_out_all_but_given_keys(self):
    stack = {'foo1': 'bar1', 'foo2': 'bar2', 'foo3': 'bar3'}
    result = stacks_view.format_stack(self.request, stack, ['foo2'])
    self.assertIn('foo2', result)
    self.assertNotIn('foo1', result)
    self.assertNotIn('foo3', result)