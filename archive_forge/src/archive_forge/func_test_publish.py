import socket
from unittest import mock
from taskflow.engines.worker_based import proxy
from taskflow import test
from taskflow.utils import threading_utils
def test_publish(self):
    msg_mock = mock.MagicMock()
    msg_data = 'msg-data'
    msg_mock.to_dict.return_value = msg_data
    routing_key = 'routing-key'
    task_uuid = 'task-uuid'
    p = self.proxy(reset_master_mock=True)
    p.publish(msg_mock, routing_key, correlation_id=task_uuid)
    mock_producer = mock.call.connection.Producer()
    master_mock_calls = self.proxy_publish_calls([mock_producer.__enter__().publish(body=msg_data, routing_key=routing_key, exchange=self.exchange_inst_mock, correlation_id=task_uuid, declare=[self.queue_inst_mock], type=msg_mock.TYPE, reply_to=None)], routing_key)
    self.master_mock.assert_has_calls(master_mock_calls)