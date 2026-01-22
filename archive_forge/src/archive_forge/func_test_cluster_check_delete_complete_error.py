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
def test_cluster_check_delete_complete_error(self):
    cluster = self._create_cluster(self.t)
    self.cl_mgr.get.side_effect = [self.fake_cl, sahara.sahara_base.APIException()]
    self.cl_mgr.get.reset_mock()
    delete_task = scheduler.TaskRunner(cluster.delete)
    ex = self.assertRaises(exception.ResourceFailure, delete_task)
    expected = 'APIException: resources.super-cluster: None'
    self.assertEqual(expected, str(ex))
    self.cl_mgr.delete.assert_called_once_with(self.fake_cl.id)
    self.assertEqual(2, self.cl_mgr.get.call_count)