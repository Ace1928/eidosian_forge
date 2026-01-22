from pprint import pformat
from six import iteritems
import re
@role_ref.setter
def role_ref(self, role_ref):
    """
        Sets the role_ref of this V1ClusterRoleBinding.
        RoleRef can only reference a ClusterRole in the global namespace. If the
        RoleRef cannot be resolved, the Authorizer must return an error.

        :param role_ref: The role_ref of this V1ClusterRoleBinding.
        :type: V1RoleRef
        """
    if role_ref is None:
        raise ValueError('Invalid value for `role_ref`, must not be `None`')
    self._role_ref = role_ref