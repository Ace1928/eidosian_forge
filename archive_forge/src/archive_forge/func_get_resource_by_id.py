from keystoneclient import exceptions as identity_exc
from keystoneclient.v3 import domains
from keystoneclient.v3 import groups
from keystoneclient.v3 import projects
from keystoneclient.v3 import users
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def get_resource_by_id(manager, resource_id):
    """Get resource by ID

    Raises CommandError if the resource is not found
    """
    try:
        return manager.get(resource_id)
    except identity_exc.NotFound:
        msg = _('Resource with id {} not found')
        raise exceptions.CommandError(msg.format(resource_id))