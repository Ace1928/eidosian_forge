import copy
from unittest import mock
from openstack.clustering.v1._proxy import Proxy
from openstack import exceptions
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import policy
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_policy_delete_not_attached(self):
    policy = self._create_policy(self.t)
    self.senlin_mock.get_policy.side_effect = [exceptions.ResourceNotFound('SenlinPolicy')]
    self.senlin_mock.detach_policy_from_cluster.side_effect = [exceptions.HttpException(http_status=400)]
    scheduler.TaskRunner(policy.delete)()
    self.senlin_mock.detach_policy_from_cluster.assert_called_once_with('c1_id', policy.resource_id)
    self.senlin_mock.delete_policy.assert_called_once_with(policy.resource_id)