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
def get_cluster_policy(self, cluster_policy, cluster):
    """Get a cluster-policy binding.

        :param cluster_policy:
            The value can be the name or ID of a policy or a
            :class:`~openstack.clustering.v1.policy.Policy` instance.
        :param cluster: The value can be the name or ID of a cluster or a
            :class:`~openstack.clustering.v1.cluster.Cluster` instance.

        :returns: a cluster-policy binding object.
        :rtype: :class:`~openstack.clustering.v1.cluster_policy.CLusterPolicy`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            cluster-policy binding matching the criteria could be found.
        """
    return self._get(_cluster_policy.ClusterPolicy, cluster_policy, cluster_id=cluster)