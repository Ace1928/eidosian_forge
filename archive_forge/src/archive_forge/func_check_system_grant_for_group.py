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
def check_system_grant_for_group(self, group_id, role_id):
    """Check if a group has a specific role on the system.

        :param group_id: the ID of the group in the assignment
        :param role_id: the ID of the system role in the assignment

        :raises keystone.exception.RoleAssignmentNotFound: if the group doesn't
            have a role assignment matching the role_id on the system

        """
    target_id = self._SYSTEM_SCOPE_TOKEN
    inherited = False
    return self.driver.check_system_grant(role_id, group_id, target_id, inherited)