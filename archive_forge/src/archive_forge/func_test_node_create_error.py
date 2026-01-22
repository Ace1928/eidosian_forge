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
def test_node_create_error(self):
    cfg.CONF.set_override('action_retry_limit', 0)
    self.senlin_mock.create_node.return_value = self.fake_node
    mock_action = mock.MagicMock()
    mock_action.status = 'FAILED'
    mock_action.status_reason = 'oops'
    self.senlin_mock.get_action.return_value = mock_action
    create_task = scheduler.TaskRunner(self.node.create)
    ex = self.assertRaises(exception.ResourceFailure, create_task)
    expected = 'ResourceInError: resources.senlin-node: Went to status FAILED due to "oops"'
    self.assertEqual(expected, str(ex))