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
def set_cluster_metadata(self, cluster, **metadata):
    """Update metadata for a cluster

        :param cluster: Either the ID of a cluster or a
            :class:`~openstack.clustering.v3.cluster.Cluster`.
        :param kwargs metadata: Key/value pairs to be updated in the cluster's
            metadata. No other metadata is modified by this call. All keys
            and values are stored as Unicode.


        :returns: A :class:`~openstack.clustering.v3.cluster.Cluster` with the
            cluster's metadata. All keys and values are Unicode text.
        :rtype: :class:`~openstack.clustering.v3.cluster.Cluster`
        """
    cluster = self._get_resource(_cluster.Cluster, cluster)
    return cluster.set_metadata(self, metadata=metadata)