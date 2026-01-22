import datetime
from unittest import mock
from oslo_config import cfg
from oslo_utils import timeutils
from heat.common import context
from heat.common import service_utils
from heat.engine import service
from heat.engine import worker
from heat.objects import service as service_objects
from heat.rpc import worker_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch('heat.common.messaging.get_rpc_client', return_value=mock.Mock())
@mock.patch('heat.common.service_utils.generate_engine_id', return_value=mock.Mock())
@mock.patch('heat.engine.service.ThreadGroupManager', return_value=mock.Mock())
@mock.patch('heat.engine.service.EngineListener', return_value=mock.Mock())
@mock.patch('heat.engine.worker.WorkerService', return_value=mock.Mock())
@mock.patch('oslo_service.threadgroup.ThreadGroup', return_value=mock.Mock())
def test_engine_service_configures_connection_pool(self, thread_group_class, worker_service_class, engine_listener_class, thread_group_manager_class, sample_uuid_method, rpc_client_class):
    self.addCleanup(self.eng._stop_rpc_server)
    self.eng.start()
    self.assertEqual(cfg.CONF.executor_thread_pool_size, cfg.CONF.database.max_overflow)