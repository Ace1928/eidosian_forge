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
def delete_profile(self, profile, ignore_missing=True):
    """Delete a profile.

        :param profile: The value can be either the name or ID of a profile or
            a :class:`~openstack.clustering.v1.profile.Profile` instance.
        :param bool ignore_missing: When set to ``False``, an exception
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the profile could not be found. When set to ``True``, no exception
            will be raised when attempting to delete a non-existent profile.

        :returns: ``None``
        """
    self._delete(_profile.Profile, profile, ignore_missing=ignore_missing)