from keystone.common.resource_options import core as ro_core
from keystone.common.validation import parameter_types
from keystone import exception
def check_immutable_update(original_resource_ref, new_resource_ref, type, resource_id):
    """Check if an update is allowed to an immutable resource.

    Valid cases where an update is allowed:

        * Resource is not immutable
        * Resource is immutable, and update to set immutable to False or None

    :param original_resource_ref: a dict resource reference representing
                                  the current resource
    :param new_resource_ref: a dict reference of the updates to perform
    :param type: the resource type, e.g. 'project'
    :param resource_id: the id of the resource (e.g. project['id']),
                        usually a UUID
    :raises: ResourceUpdateForbidden
    """
    immutable = check_resource_immutable(original_resource_ref)
    if immutable:
        new_options = new_resource_ref.get('options', {})
        if len(new_resource_ref.keys()) > 1 or IMMUTABLE_OPT.option_name not in new_options or new_options[IMMUTABLE_OPT.option_name] not in (False, None):
            raise exception.ResourceUpdateForbidden(type=type, resource_id=resource_id)