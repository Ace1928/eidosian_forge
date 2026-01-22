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
@mock.patch.object(service_objects.Service, 'get_all_by_args')
@mock.patch.object(service_objects.Service, 'delete')
@mock.patch.object(context, 'get_admin_context')
def test_service_manage_report_cleanup(self, mock_admin_context, mock_service_delete, mock_get_all):
    mock_admin_context.return_value = self.ctx
    ages_a_go = timeutils.utcnow() - datetime.timedelta(seconds=4000)
    mock_get_all.return_value = [{'id': 'foo', 'deleted_at': None, 'updated_at': ages_a_go}]
    self.eng.service_manage_cleanup()
    mock_admin_context.assert_called_once_with()
    mock_get_all.assert_called_once_with(self.ctx, self.eng.host, self.eng.binary, self.eng.hostname)
    mock_service_delete.assert_called_once_with(self.ctx, 'foo')