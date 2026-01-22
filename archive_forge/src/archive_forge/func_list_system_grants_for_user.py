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
def list_system_grants_for_user(self, user_id):
    """Return a list of roles the user has on the system.

        :param user_id: the ID of the user

        :returns: a list of role assignments the user has system-wide

        """
    target_id = self._SYSTEM_SCOPE_TOKEN
    assignment_type = self._USER_SYSTEM
    grants = self.driver.list_system_grants(user_id, target_id, assignment_type)
    grant_ids = []
    for grant in grants:
        grant_ids.append(grant['role_id'])
    return PROVIDERS.role_api.list_roles_from_ids(grant_ids)