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
def test_delete_not_found_during_delete(self):
    tc = self._create_resource('cluster', self.rsrc_defn, self.stack)
    fake_cluster = FakeTroveCluster()
    fake_cluster.task = {'name': 'NONE'}
    fake_cluster.delete = mock.Mock(side_effect=[troveexc.NotFound()])
    self.client.clusters.get.side_effect = [fake_cluster, fake_cluster, troveexc.NotFound()]
    scheduler.TaskRunner(tc.delete)()
    self.assertEqual((tc.DELETE, tc.COMPLETE), tc.state)
    self.assertEqual(1, fake_cluster.delete.call_count)