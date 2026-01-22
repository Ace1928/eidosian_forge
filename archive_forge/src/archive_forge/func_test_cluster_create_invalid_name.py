from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import cluster as sc
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_cluster_create_invalid_name(self):
    cluster = self._init_cluster(self.t2, 'lots_of_underscore_name')
    self.cl_mgr.create.return_value = self.fake_cl
    self.cl_mgr.get.return_value = self.fake_cl
    scheduler.TaskRunner(cluster.create)()
    name = self.cl_mgr.create.call_args[0][0]
    self.assertIn('lotsofunderscorename', name)