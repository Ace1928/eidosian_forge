from unittest import mock
from heat.db import api as db_api
from heat.engine import check_resource
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.engine import worker
from heat.objects import stack as stack_objects
from heat.rpc import worker_client as wc
from heat.tests import common
from heat.tests import utils
def test_service_stop(self):
    self.worker = worker.WorkerService('host-1', 'topic-1', 'engine_id', mock.Mock())
    with mock.patch.object(self.worker, '_rpc_server') as mock_rpc_server:
        self.worker.stop()
        mock_rpc_server.stop.assert_called_once_with()
        mock_rpc_server.wait.assert_called_once_with()