from unittest import mock
from openstack.clustering.v1 import _proxy
from openstack.clustering.v1 import action
from openstack.clustering.v1 import build_info
from openstack.clustering.v1 import cluster
from openstack.clustering.v1 import cluster_attr
from openstack.clustering.v1 import cluster_policy
from openstack.clustering.v1 import event
from openstack.clustering.v1 import node
from openstack.clustering.v1 import policy
from openstack.clustering.v1 import policy_type
from openstack.clustering.v1 import profile
from openstack.clustering.v1 import profile_type
from openstack.clustering.v1 import receiver
from openstack.clustering.v1 import service
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
def test_set_cluster_metadata(self):
    kwargs = {'a': '1', 'b': '2'}
    id = 'an_id'
    self._verify('openstack.clustering.v1.cluster.Cluster.set_metadata', self.proxy.set_cluster_metadata, method_args=[id], method_kwargs=kwargs, method_result=cluster.Cluster.existing(id=id, metadata=kwargs), expected_args=[self.proxy], expected_kwargs={'metadata': kwargs}, expected_result=cluster.Cluster.existing(id=id, metadata=kwargs))