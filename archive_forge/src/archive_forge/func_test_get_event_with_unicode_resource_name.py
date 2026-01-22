from unittest import mock
import testtools
from heatclient.common import utils
from heatclient.v1 import events
@mock.patch('heatclient.v1.events.EventManager._resolve_stack_id')
@mock.patch('heatclient.common.utils.get_response_body')
def test_get_event_with_unicode_resource_name(self, mock_utils, mock_re):
    fields = {'stack_id': 'teststack', 'resource_name': '工作', 'event_id': '1'}

    class FakeAPI(object):
        """Fake API and ensure request url is correct."""

        def json_request(self, *args, **kwargs):
            expect = ('GET', '/stacks/teststack/abcd1234/resources/%E5%B7%A5%E4%BD%9C/events/1')
            assert args == expect
            return ({}, {'event': []})

        def get(self, *args, **kwargs):
            pass
    manager = events.EventManager(FakeAPI())
    with mock.patch('heatclient.v1.events.Event'):
        mock_utils.return_value = {'event': []}
        mock_re.return_value = 'teststack/abcd1234'
        manager.get(**fields)
        mock_re.assert_called_once_with('teststack')