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
def get_roles_for_trustor_and_project(self, trustor_id, project_id):
    """Get the roles associated with a trustor within given project.

        This includes roles directly assigned to the trustor on the
        project, as well as those by virtue of group membership or
        inheritance, but it doesn't include the domain roles.

        :returns: a list of role ids.
        :raises keystone.exception.ProjectNotFound: If the project doesn't
            exist.

        """
    PROVIDERS.resource_api.get_project(project_id)
    assignment_list = self.list_role_assignments(user_id=trustor_id, project_id=project_id, effective=True, strip_domain_roles=False)
    return list(set([x['role_id'] for x in assignment_list]))