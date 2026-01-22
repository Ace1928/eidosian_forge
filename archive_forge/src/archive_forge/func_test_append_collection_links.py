from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
@mock.patch.object(stacks_view.views_common, 'get_collection_links')
def test_append_collection_links(self, mock_get_collection_links):
    stacks = [self.stack1]
    mock_get_collection_links.return_value = 'fake links'
    stack_view = stacks_view.collection(self.request, stacks)
    self.assertIn('links', stack_view)