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
@mock.patch.object(service_objects.Service, 'update_by_id')
@mock.patch.object(context, 'get_admin_context')
def test_service_manage_report_update(self, mock_admin_context, mock_service_update):
    self.eng.service_id = 'mock_id'
    mock_admin_context.return_value = self.ctx
    self.eng.service_manage_report()
    mock_admin_context.assert_called_once_with()
    mock_service_update.assert_called_once_with(self.ctx, 'mock_id', dict(deleted_at=None))