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
def test_policy_resolve_attribute(self):
    excepted_show = {'id': 'some_id', 'name': 'SenlinPolicy'}
    policy = self._create_policy(self.t)
    self.senlin_mock.get_policy.return_value = FakePolicy()
    self.assertEqual(excepted_show, policy._show_resource())