from openstack.clustering.v1 import action as _action
from openstack.clustering.v1 import build_info
from openstack.clustering.v1 import cluster as _cluster
from openstack.clustering.v1 import cluster_attr as _cluster_attr
from openstack.clustering.v1 import cluster_policy as _cluster_policy
from openstack.clustering.v1 import event as _event
from openstack.clustering.v1 import node as _node
from openstack.clustering.v1 import policy as _policy
from openstack.clustering.v1 import policy_type as _policy_type
from openstack.clustering.v1 import profile as _profile
from openstack.clustering.v1 import profile_type as _profile_type
from openstack.clustering.v1 import receiver as _receiver
from openstack.clustering.v1 import service as _service
from openstack import proxy
from openstack import resource
def cluster_policies(self, cluster, **query):
    """Retrieve a generator of cluster-policy bindings.

        :param cluster: The value can be the name or ID of a cluster or a
            :class:`~openstack.clustering.v1.cluster.Cluster` instance.
        :param kwargs query: Optional query parameters to be sent to
            restrict the policies to be returned. Available parameters include:

            * enabled: A boolean value indicating whether the policy is
              enabled on the cluster.
        :returns: A generator of cluster-policy binding instances.
        """
    cluster_id = resource.Resource._get_id(cluster)
    return self._list(_cluster_policy.ClusterPolicy, cluster_id=cluster_id, **query)