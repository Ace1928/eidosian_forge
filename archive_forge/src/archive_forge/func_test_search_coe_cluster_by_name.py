from openstack.container_infrastructure_management.v1 import cluster
from openstack.tests.unit import base
def test_search_coe_cluster_by_name(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clusters'), json=dict(clusters=[coe_cluster_obj]))])
    coe_clusters = self.cloud.search_coe_clusters(name_or_id='k8s')
    self.assertEqual(1, len(coe_clusters))
    self.assertEqual(coe_cluster_obj['uuid'], coe_clusters[0]['id'])
    self.assert_calls()