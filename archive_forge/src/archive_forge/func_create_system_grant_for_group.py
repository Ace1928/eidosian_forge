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
def create_system_grant_for_group(self, group_id, role_id):
    """Grant a group a role on the system.

        :param group_id: the ID of the group
        :param role_id: the ID of the role to grant on the system

        """
    role = PROVIDERS.role_api.get_role(role_id)
    if role.get('domain_id'):
        raise exception.ValidationError('Role %(role_id)s is a domain-specific role. Unable to use a domain-specific role in a system assignment.' % {'role_id': role_id})
    target_id = self._SYSTEM_SCOPE_TOKEN
    assignment_type = self._GROUP_SYSTEM
    inherited = False
    self.driver.create_system_grant(role_id, group_id, target_id, assignment_type, inherited)