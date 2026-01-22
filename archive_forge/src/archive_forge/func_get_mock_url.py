import testtools
from openstack.container_infrastructure_management.v1 import cluster_template
from openstack import exceptions
from openstack.tests.unit import base
def get_mock_url(self, service_type='container-infrastructure-management', base_url_append=None, append=None, resource=None):
    return super(TestClusterTemplates, self).get_mock_url(service_type=service_type, resource=resource, append=append, base_url_append=base_url_append)