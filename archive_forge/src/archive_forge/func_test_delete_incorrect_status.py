import copy
from unittest import mock
from troveclient import exceptions as troveexc
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import trove
from heat.engine.resources.openstack.trove import cluster
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_delete_incorrect_status(self):
    tc = self._create_resource('cluster', self.rsrc_defn, self.stack)
    fake_cluster_bad = FakeTroveCluster()
    fake_cluster_bad.task = {'name': 'BUILDING'}
    fake_cluster_bad.delete = mock.Mock()
    fake_cluster_ok = FakeTroveCluster()
    fake_cluster_ok.task = {'name': 'NONE'}
    fake_cluster_ok.delete = mock.Mock()
    self.client.clusters.get.side_effect = [fake_cluster_bad, fake_cluster_bad, fake_cluster_ok, troveexc.NotFound()]
    scheduler.TaskRunner(tc.delete)()
    self.assertEqual((tc.DELETE, tc.COMPLETE), tc.state)
    fake_cluster_bad.delete.assert_not_called()
    fake_cluster_ok.delete.assert_called_once_with()