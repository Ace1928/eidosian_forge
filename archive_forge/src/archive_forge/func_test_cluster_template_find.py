from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import _proxy
from openstack.container_infrastructure_management.v1 import cluster
from openstack.container_infrastructure_management.v1 import cluster_template
from openstack.container_infrastructure_management.v1 import service
from openstack.tests.unit import test_proxy_base
def test_cluster_template_find(self):
    self.verify_find(self.proxy.find_cluster_template, cluster_template.ClusterTemplate, method_kwargs={}, expected_kwargs={})