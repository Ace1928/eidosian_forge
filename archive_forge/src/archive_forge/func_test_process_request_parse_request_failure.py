from taskflow.engines.worker_based import endpoint as ep
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import server
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
@mock.patch.object(failure.Failure, 'from_dict')
@mock.patch.object(failure.Failure, 'to_dict')
def test_process_request_parse_request_failure(self, to_mock, from_mock):
    failure_dict = {'failure': 'failure'}
    a_failure = failure.Failure.from_exception(RuntimeError('Woot!'))
    to_mock.return_value = failure_dict
    from_mock.side_effect = ValueError('Woot!')
    request = self.make_request(result=a_failure)
    s = self.server(reset_master_mock=True)
    s._process_request(request, self.message_mock)
    master_mock_calls = [mock.call.Response(pr.FAILURE, result=failure_dict), mock.call.proxy.publish(self.response_inst_mock, self.reply_to, correlation_id=self.task_uuid)]
    self.master_mock.assert_has_calls(master_mock_calls)