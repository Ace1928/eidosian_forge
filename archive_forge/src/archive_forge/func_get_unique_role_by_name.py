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
def get_unique_role_by_name(self, role_name, hints=None):
    if not hints:
        hints = driver_hints.Hints()
    hints.add_filter('name', role_name, case_sensitive=True)
    found_roles = PROVIDERS.role_api.list_roles(hints)
    if not found_roles:
        raise exception.RoleNotFound(_('Role %s is not defined') % role_name)
    elif len(found_roles) == 1:
        return {'id': found_roles[0]['id']}
    else:
        raise exception.AmbiguityError(resource='role', name=role_name)