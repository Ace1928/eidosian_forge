from unittest import mock
import testtools
from heatclient.common import event_utils
from heatclient.v1 import events as hc_ev
from heatclient.v1 import resources as hc_res
def test_get_nested_events(self):
    resources = {'parent': self._mock_resource('resource1', 'foo/child1'), 'foo/child1': self._mock_resource('res_child1', 'foo/child2'), 'foo/child2': self._mock_resource('res_child2', 'foo/child3'), 'foo/child3': self._mock_resource('res_child3', 'foo/END')}

    def resource_list_stub(stack_id):
        return [resources[stack_id]]
    mock_client = mock.MagicMock()
    mock_client.resources.list.side_effect = resource_list_stub
    events = {'foo/child1': self._mock_event('event1', 'res_child1'), 'foo/child2': self._mock_event('event2', 'res_child2'), 'foo/child3': self._mock_event('event3', 'res_child3')}

    def event_list_stub(stack_id, argfoo):
        return [events[stack_id]]
    mock_client.events.list.side_effect = event_list_stub
    ev_args = {'argfoo': 123}
    evs = event_utils._get_nested_events(hc=mock_client, nested_depth=1, stack_id='parent', event_args=ev_args)
    rsrc_calls = [mock.call(stack_id='parent')]
    mock_client.resources.list.assert_has_calls(rsrc_calls)
    ev_calls = [mock.call(stack_id='foo/child1', argfoo=123)]
    mock_client.events.list.assert_has_calls(ev_calls)
    self.assertEqual(1, len(evs))
    self.assertEqual('event1', evs[0].id)
    mock_client.resources.list.reset_mock()
    mock_client.events.list.reset_mock()
    evs = event_utils._get_nested_events(hc=mock_client, nested_depth=3, stack_id='parent', event_args=ev_args)
    rsrc_calls = [mock.call(stack_id='parent'), mock.call(stack_id='foo/child1'), mock.call(stack_id='foo/child2')]
    mock_client.resources.list.assert_has_calls(rsrc_calls)
    ev_calls = [mock.call(stack_id='foo/child1', argfoo=123), mock.call(stack_id='foo/child2', argfoo=123), mock.call(stack_id='foo/child3', argfoo=123)]
    mock_client.events.list.assert_has_calls(ev_calls)
    self.assertEqual(3, len(evs))
    self.assertEqual('event1', evs[0].id)
    self.assertEqual('event2', evs[1].id)
    self.assertEqual('event3', evs[2].id)