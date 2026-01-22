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
def list_all_system_grants(self):
    """Return a list of all system grants."""
    actor_id = None
    target_id = self._SYSTEM_SCOPE_TOKEN
    assignment_type = None
    return self.driver.list_system_grants(actor_id, target_id, assignment_type)