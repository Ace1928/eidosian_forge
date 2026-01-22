from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
def test_includes_all_other_keys(self):
    stack = {'foo': 'bar'}
    result = stacks_view.format_stack(self.request, stack)
    self.assertIn('foo', result)
    self.assertEqual('bar', result['foo'])