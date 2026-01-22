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
def list_domains_for_user(self, user_id):
    assignment_list = self.list_role_assignments(user_id=user_id, effective=True)
    domain_ids = list(set([x['domain_id'] for x in assignment_list if x.get('domain_id')]))
    return PROVIDERS.resource_api.list_domains_from_ids(domain_ids)