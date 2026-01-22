from taskflow.engines.worker_based import endpoint as ep
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import server
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
@mock.patch.object(failure.Failure, 'to_dict')
def test_process_request_task_failure(self, to_mock):
    failure_dict = {'failure': 'failure'}
    to_mock.return_value = failure_dict
    request = self.make_request(task=utils.TaskWithFailure(), arguments={})
    s = self.server(reset_master_mock=True)
    s._process_request(request, self.message_mock)
    master_mock_calls = [mock.call.Response(pr.RUNNING), mock.call.proxy.publish(self.response_inst_mock, self.reply_to, correlation_id=self.task_uuid), mock.call.Response(pr.FAILURE, result=failure_dict), mock.call.proxy.publish(self.response_inst_mock, self.reply_to, correlation_id=self.task_uuid)]
    self.master_mock.assert_has_calls(master_mock_calls)