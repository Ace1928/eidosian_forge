from unittest import mock
import testtools
from heatclient.common import utils
from heatclient.v1 import events
def test_list_event_with_unicode_resource_name(self):
    stack_id = ('teststack',)
    resource_name = '工作'
    manager = events.EventManager(None)
    with mock.patch('heatclient.v1.events.EventManager._resolve_stack_id') as mock_re:
        mock_re.return_value = 'teststack/abcd1234'
        manager._list = mock.MagicMock()
        manager.list(stack_id, resource_name)
        manager._list.assert_called_once_with('/stacks/teststack/abcd1234/resources/%E5%B7%A5%E4%BD%9C/events', 'events')
        mock_re.assert_called_once_with(stack_id)