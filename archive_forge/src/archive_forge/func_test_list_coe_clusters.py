from openstack.container_infrastructure_management.v1 import cluster
from openstack.tests.unit import base
def test_list_coe_clusters(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clusters'), json=dict(clusters=[coe_cluster_obj]))])
    cluster_list = self.cloud.list_coe_clusters()
    self._compare_clusters(coe_cluster_obj, cluster_list[0])
    self.assert_calls()