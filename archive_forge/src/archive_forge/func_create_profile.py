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
def create_profile(self, **attrs):
    """Create a new profile from attributes.

        :param dict attrs: Keyword arguments that will be used to create a
            :class:`~openstack.clustering.v1.profile.Profile`, it is comprised
            of the properties on the Profile class.

        :returns: The results of profile creation.
        :rtype: :class:`~openstack.clustering.v1.profile.Profile`.
        """
    return self._create(_profile.Profile, **attrs)