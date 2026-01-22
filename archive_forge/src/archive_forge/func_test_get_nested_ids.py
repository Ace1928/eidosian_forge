from unittest import mock
import testtools
from heatclient.common import event_utils
from heatclient.v1 import events as hc_ev
from heatclient.v1 import resources as hc_res
def test_get_nested_ids(self):

    def list_stub(stack_id):
        return [self._mock_resource('aresource', 'foo3/3id')]
    mock_client = mock.MagicMock()
    mock_client.resources.list.side_effect = list_stub
    ids = event_utils._get_nested_ids(hc=mock_client, stack_id='astack/123')
    mock_client.resources.list.assert_called_once_with(stack_id='astack/123')
    self.assertEqual(['foo3/3id'], ids)