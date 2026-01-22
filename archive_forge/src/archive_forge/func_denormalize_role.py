from keystone.assignment.backends import base
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
def denormalize_role(ref):
    assignment = {}
    if ref.type == AssignmentType.USER_PROJECT:
        assignment['user_id'] = ref.actor_id
        assignment['project_id'] = ref.target_id
    elif ref.type == AssignmentType.USER_DOMAIN:
        assignment['user_id'] = ref.actor_id
        assignment['domain_id'] = ref.target_id
    elif ref.type == AssignmentType.GROUP_PROJECT:
        assignment['group_id'] = ref.actor_id
        assignment['project_id'] = ref.target_id
    elif ref.type == AssignmentType.GROUP_DOMAIN:
        assignment['group_id'] = ref.actor_id
        assignment['domain_id'] = ref.target_id
    else:
        raise exception.Error(message=_('Unexpected assignment type encountered, %s') % ref.type)
    assignment['role_id'] = ref.role_id
    if ref.inherited:
        assignment['inherited_to_projects'] = 'projects'
    return assignment