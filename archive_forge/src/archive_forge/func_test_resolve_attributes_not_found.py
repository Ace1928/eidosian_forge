from unittest import mock
from blazarclient import exception as client_exception
from oslo_utils.fixture import uuidsentinel as uuids
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import blazar
from heat.engine.resources.openstack.blazar import host
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_resolve_attributes_not_found(self):
    host_resource = self._create_resource('host', self.rsrc_defn, self.stack)
    scheduler.TaskRunner(host_resource.create)()
    self.client.host.get.return_value = self.host
    self.assertRaises(exception.InvalidTemplateAttribute, host_resource._resolve_attribute, 'invalid_attribute')