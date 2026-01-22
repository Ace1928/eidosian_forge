import io
from oslo_utils import reflection
from taskflow.engines.worker_based import endpoint
from taskflow.engines.worker_based import worker
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
def test_creation_with_custom_executor(self):
    executor_mock = mock.MagicMock(name='executor')
    self.worker(executor=executor_mock)
    master_mock_calls = [mock.call.Server(self.topic, self.exchange, executor_mock, [], url=self.broker_url, transport_options=mock.ANY, transport=mock.ANY, retry_options=mock.ANY)]
    self.assertEqual(master_mock_calls, self.master_mock.mock_calls)