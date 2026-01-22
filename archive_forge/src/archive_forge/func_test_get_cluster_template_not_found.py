import testtools
from openstack.container_infrastructure_management.v1 import cluster_template
from openstack import exceptions
from openstack.tests.unit import base
def test_get_cluster_template_not_found(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[]))])
    r = self.cloud.get_cluster_template('doesNotExist')
    self.assertIsNone(r)
    self.assert_calls()