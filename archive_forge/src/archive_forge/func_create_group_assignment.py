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
def create_group_assignment(base_ref, user_id):
    """Create a group assignment from the provided ref."""
    ref = copy.deepcopy(base_ref)
    ref['user_id'] = user_id
    indirect = ref.setdefault('indirect', {})
    indirect['group_id'] = ref.pop('group_id')
    return ref