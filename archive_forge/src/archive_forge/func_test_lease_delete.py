from unittest import mock
from blazarclient import exception as client_exception
from oslo_utils.fixture import uuidsentinel as uuids
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import blazar
from heat.engine.resources.openstack.blazar import lease
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_lease_delete(self):
    lease_resource = self._create_resource('lease', self.rsrc_defn, self.stack)
    self.client.lease.delete.return_value = None
    scheduler.TaskRunner(lease_resource.create)()
    self.client.lease.get.side_effect = ['lease_obj', client_exception.BlazarClientException(code=404)]
    scheduler.TaskRunner(lease_resource.delete)()
    self.assertEqual((lease_resource.DELETE, lease_resource.COMPLETE), lease_resource.state)
    self.assertEqual(1, self.client.lease.delete.call_count)