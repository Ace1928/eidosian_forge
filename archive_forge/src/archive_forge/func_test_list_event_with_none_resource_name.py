from unittest import mock
import testtools
from heatclient.common import utils
from heatclient.v1 import events
def test_list_event_with_none_resource_name(self):
    stack_id = ('teststack',)
    manager = events.EventManager(None)
    manager._list = mock.MagicMock()
    manager.list(stack_id)
    manager._list.assert_called_once_with('/stacks/teststack/events', 'events')