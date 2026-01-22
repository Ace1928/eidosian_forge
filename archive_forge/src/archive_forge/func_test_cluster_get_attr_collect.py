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
def test_cluster_get_attr_collect(self):
    cluster = self._create_cluster(self.t)
    self.senlin_mock.collect_cluster_attrs.return_value = [mock.Mock(attr_value='ip1')]
    attr_path1 = ['details.addresses.private[0].addr']
    self.assertEqual(['ip1'], cluster.get_attribute(cluster.ATTR_COLLECT, *attr_path1))
    attr_path2 = ['details.addresses.private[0].addr', 0]
    self.assertEqual('ip1', cluster.get_attribute(cluster.ATTR_COLLECT, *attr_path2))
    self.senlin_mock.collect_cluster_attrs.assert_called_with(cluster.resource_id, attr_path2[0])