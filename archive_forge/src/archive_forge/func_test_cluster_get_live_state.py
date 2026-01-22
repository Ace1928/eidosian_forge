import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import cluster as sc
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
from openstack import exceptions
def test_cluster_get_live_state(self):
    expected_reality = {'name': 'SenlinCluster', 'metadata': {'foo': 'bar'}, 'timeout': 3600, 'desired_capacity': 1, 'max_size': -1, 'min_size': 0, 'profile': 'fake_profile_id', 'policies': [{'policy': 'fake_policy_id', 'enabled': True}]}
    cluster = self._create_cluster(self.t)
    self.senlin_mock.get_cluster.return_value = self.fake_cl
    reality = cluster.get_live_state(cluster.properties)
    self.assertEqual(expected_reality, reality)