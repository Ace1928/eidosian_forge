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
@MEMOIZE_COMPUTED_ASSIGNMENTS
def get_roles_for_user_and_domain(self, user_id, domain_id):
    """Get the roles associated with a user within given domain.

        :returns: a list of role ids.
        :raises keystone.exception.DomainNotFound: If the domain doesn't exist.

        """
    PROVIDERS.resource_api.get_domain(domain_id)
    assignment_list = self.list_role_assignments(user_id=user_id, domain_id=domain_id, effective=True)
    return list(set([x['role_id'] for x in assignment_list]))