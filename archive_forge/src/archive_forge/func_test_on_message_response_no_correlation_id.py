import threading
import time
from taskflow.engines.worker_based import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_on_message_response_no_correlation_id(self):
    self.message_mock.properties = {'type': pr.RESPONSE}
    response = pr.Response(pr.RUNNING)
    ex = self.executor()
    ex._ongoing_requests[self.task_uuid] = self.request_inst_mock
    ex._process_response(response.to_dict(), self.message_mock)
    self.assertEqual([], self.request_inst_mock.mock_calls)