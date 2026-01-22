import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_create_with_discovery_url(self):
    cluster_with_discovery = dict()
    cluster_with_discovery.update(CREATE_CLUSTER)
    cluster_with_discovery['discovery_url'] = 'discovery_url'
    cluster = self.mgr.create(**cluster_with_discovery)
    expect = [('POST', '/v1/clusters', {}, cluster_with_discovery)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(cluster)