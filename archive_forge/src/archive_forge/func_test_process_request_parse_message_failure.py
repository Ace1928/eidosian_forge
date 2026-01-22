from taskflow.engines.worker_based import endpoint as ep
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import server
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
@mock.patch('taskflow.engines.worker_based.server.LOG.warning')
def test_process_request_parse_message_failure(self, mocked_exception):
    self.message_mock.properties = {}
    request = self.make_request()
    s = self.server(reset_master_mock=True)
    s._process_request(request, self.message_mock)
    self.assertTrue(mocked_exception.called)