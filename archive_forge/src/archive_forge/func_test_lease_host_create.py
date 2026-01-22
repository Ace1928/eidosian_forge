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
def test_lease_host_create(self):
    self.patchobject(blazar.BlazarClientPlugin, 'client', return_value=self.client)
    self.client.has_host.return_value = True
    lease_resource = self._create_resource('lease', self.rsrc_defn, self.stack)
    self.assertEqual(self.lease['name'], lease_resource.properties.get(lease.Lease.NAME))
    self.assertIsNone(lease_resource.validate())
    scheduler.TaskRunner(lease_resource.create)()
    self.assertEqual(uuids.lease_id, lease_resource.resource_id)
    self.assertEqual((lease_resource.CREATE, lease_resource.COMPLETE), lease_resource.state)
    self.assertEqual('lease', lease_resource.entity)
    self.client.lease.create.assert_called_once_with(name=self.lease['name'], start=self.lease['start_date'], end=self.lease['end_date'], reservations=self.lease['reservations'], events=self.lease['events'])