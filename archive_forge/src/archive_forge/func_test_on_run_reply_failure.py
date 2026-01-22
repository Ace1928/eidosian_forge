from taskflow.engines.worker_based import endpoint as ep
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import server
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def test_on_run_reply_failure(self):
    request = self.make_request(task=utils.ProgressingTask(), arguments={})
    self.proxy_inst_mock.publish.side_effect = RuntimeError('Woot!')
    s = self.server(reset_master_mock=True)
    s._process_request(request, self.message_mock)
    self.assertEqual(1, self.proxy_inst_mock.publish.call_count)