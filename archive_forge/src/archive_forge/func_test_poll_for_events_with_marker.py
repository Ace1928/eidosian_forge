from unittest import mock
import testtools
from heatclient.common import event_utils
from heatclient.v1 import events as hc_ev
from heatclient.v1 import resources as hc_res
@mock.patch('heatclient.common.event_utils.get_events')
def test_poll_for_events_with_marker(self, ge):
    ge.side_effect = [[self._mock_event('5', 'res_child1', 'CREATE_COMPLETE'), self._mock_event('6', 'res_child2', 'CREATE_COMPLETE'), self._mock_event('7', 'res_child3', 'CREATE_COMPLETE'), self._mock_stack_event('8', 'astack', 'CREATE_COMPLETE')]]
    stack_status, msg = event_utils.poll_for_events(None, 'astack', action='CREATE', poll_period=0, marker='4', nested_depth=0)
    self.assertEqual('CREATE_COMPLETE', stack_status)
    self.assertEqual('\n Stack astack CREATE_COMPLETE \n', msg)
    ge.assert_has_calls([mock.call(None, stack_id='astack', nested_depth=0, event_args={'sort_dir': 'asc', 'marker': '4'})])