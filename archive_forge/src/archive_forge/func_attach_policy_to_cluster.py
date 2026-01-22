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
def attach_policy_to_cluster(self, cluster, policy, **params):
    """Attach a policy to a cluster.

        :param cluster: Either the name or the ID of the cluster, or an
            instance of :class:`~openstack.clustering.v1.cluster.Cluster`.
        :param policy: Either the name or the ID of a policy.
        :param dict params: A dictionary containing the properties for the
            policy to be attached.
        :returns: A dict containing the action initiated by this operation.
        """
    if isinstance(cluster, _cluster.Cluster):
        obj = cluster
    else:
        obj = self._find(_cluster.Cluster, cluster, ignore_missing=False)
    return obj.policy_attach(self, policy, **params)