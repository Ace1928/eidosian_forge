from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
@mock.patch.object(stacks_view, 'util')
def test_replace_stack_identity_with_id_and_links(self, mock_util):
    mock_util.make_link.return_value = 'blah'
    stack = {'stack_identity': {'stack_id': 'foo'}}
    result = stacks_view.format_stack(self.request, stack)
    self.assertIn('id', result)
    self.assertNotIn('stack_identity', result)
    self.assertEqual('foo', result['id'])
    self.assertIn('links', result)
    self.assertEqual(['blah'], result['links'])