import testtools
from openstack.container_infrastructure_management.v1 import cluster_template
from openstack import exceptions
from openstack.tests.unit import base
def test_create_cluster_template_exception(self):
    self.register_uris([dict(method='POST', uri=self.get_mock_url(resource='clustertemplates'), status_code=403)])
    with testtools.ExpectedException(exceptions.ForbiddenException):
        self.cloud.create_cluster_template('fake-cluster-template')
    self.assert_calls()