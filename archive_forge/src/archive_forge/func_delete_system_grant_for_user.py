import copy
import itertools
from oslo_log import log
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def delete_system_grant_for_user(self, user_id, role_id):
    """Remove a system grant from a user.

        :param user_id: the ID of the user
        :param role_id: the ID of the role to remove from the user on the
                        system

        :raises keystone.exception.RoleAssignmentNotFound: if the user doesn't
            have a role assignment with role_id on the system

        """
    target_id = self._SYSTEM_SCOPE_TOKEN
    inherited = False
    self.driver.delete_system_grant(role_id, user_id, target_id, inherited)
    COMPUTED_ASSIGNMENTS_REGION.invalidate()