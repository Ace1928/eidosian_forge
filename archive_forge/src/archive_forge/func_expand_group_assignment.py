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
def expand_group_assignment(ref, user_id):
    """Expand group role assignment.

            For any group role assignment on a target, it is replaced by a list
            of role assignments containing one for each user of that group on
            that target.

            An example of accepted ref is::

            {
                'group_id': group_id,
                'project_id': project_id,
                'role_id': role_id
            }

            Once expanded, it should be returned as a list of entities like the
            one below, one for each user_id in the provided group_id.

            ::

            {
                'user_id': user_id,
                'project_id': project_id,
                'role_id': role_id,
                'indirect' : {
                    'group_id': group_id
                }
            }

            Returned list will be formatted by the Controller, which will
            deduce a role assignment came from group membership if it has both
            'user_id' in the main body of the dict and 'group_id' in indirect
            subdict.

            """
    if user_id:
        return [create_group_assignment(ref, user_id=user_id)]
    try:
        users = PROVIDERS.identity_api.list_users_in_group(ref['group_id'])
    except exception.GroupNotFound:
        LOG.warning('Group %(group)s was not found but still has role assignments.', {'group': ref['group_id']})
        users = []
    return [create_group_assignment(ref, user_id=m['id']) for m in users]