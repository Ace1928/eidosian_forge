from unittest import mock
import testtools
from heatclient.common import utils
from heatclient.v1 import events
def test_list_event_with_kwargs(self):
    stack_id = ('teststack',)
    resource_name = 'testresource'
    kwargs = {'limit': 2, 'marker': '6d6935f4-0ae5', 'filters': {'resource_action': 'CREATE', 'resource_status': 'COMPLETE'}}
    manager = events.EventManager(None)
    manager = events.EventManager(None)
    with mock.patch('heatclient.v1.events.EventManager._resolve_stack_id') as mock_re:
        mock_re.return_value = 'teststack/abcd1234'
        manager._list = mock.MagicMock()
        manager.list(stack_id, resource_name, **kwargs)
        self.assertEqual(1, manager._list.call_count)
        args = manager._list.call_args
        self.assertEqual(2, len(args[0]))
        url, param = args[0]
        self.assertEqual('events', param)
        base_url, query_params = utils.parse_query_url(url)
        expected_base_url = '/stacks/teststack/abcd1234/resources/testresource/events'
        self.assertEqual(expected_base_url, base_url)
        expected_query_dict = {'marker': ['6d6935f4-0ae5'], 'limit': ['2'], 'resource_action': ['CREATE'], 'resource_status': ['COMPLETE']}
        self.assertEqual(expected_query_dict, query_params)
        mock_re.assert_called_once_with(stack_id)