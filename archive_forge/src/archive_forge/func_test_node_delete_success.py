import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import node as sn
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
from openstack import exceptions
def test_node_delete_success(self):
    node = self._create_node()
    self.senlin_mock.get_node.side_effect = [exceptions.ResourceNotFound('SenlinNode')]
    scheduler.TaskRunner(node.delete)()
    self.senlin_mock.delete_node.assert_called_once_with(node.resource_id)