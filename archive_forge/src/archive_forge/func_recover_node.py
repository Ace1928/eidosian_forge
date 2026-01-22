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
def recover_node(self, node, **params):
    """Recover the specified node into healthy status.

        :param node: The value can be either the ID of a node or a
            :class:`~openstack.clustering.v1.node.Node` instance.
        :param dict params: A dict supplying parameters to the recover action.

        :returns: A dictionary containing the action ID.
        """
    obj = self._get_resource(_node.Node, node)
    return obj.recover(self, **params)