from unittest import mock
from oslo_metrics import message_router
from oslotest import base
import prometheus_client
def test_process_histogram(self):
    received_json = '{\n  "module": "oslo_messaging",\n  "name": "rpc_client_processing_seconds",\n  "action": {\n    "action": "observe",\n    "value": 1.26\n  },\n  "labels": {\n    "call_type": "call",\n    "exchange": "foo",\n    "topic": "bar",\n    "method": "get",\n    "server": "foobar",\n    "namespace": "ns",\n    "version": "v2",\n    "process": "done",\n    "fanout": "foo",\n    "timeout": 10\n  }\n}'.encode()
    with mock.patch.object(prometheus_client.Histogram, 'observe') as mock_inc:
        router = message_router.MessageRouter()
        router.process(received_json)
        mock_inc.assert_called_once_with(1.26)