import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_show_by_id(self):
    cluster = self.mgr.get(CLUSTER1['id'])
    expect = [('GET', '/v1/clusters/%s' % CLUSTER1['id'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(CLUSTER1['name'], cluster.name)
    self.assertEqual(CLUSTER1['cluster_template_id'], cluster.cluster_template_id)