from keystoneclient import exceptions as identity_exc
from keystoneclient.v3 import domains
from keystoneclient.v3 import groups
from keystoneclient.v3 import projects
from keystoneclient.v3 import users
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def find_service_in_list(service_list, service_id):
    """Find a service by id in service list."""
    for service in service_list:
        if service.id == service_id:
            return service
    raise exceptions.CommandError("No service with a type, name or ID of '%s' exists." % service_id)