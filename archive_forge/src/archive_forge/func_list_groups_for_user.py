import uuid
import ldap.filter
from oslo_log import log
from oslo_log import versionutils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import base
from keystone.identity.backends.ldap import common as common_ldap
from keystone.identity.backends.ldap import models
def list_groups_for_user(self, user_id, hints):
    user_ref = self._get_user(user_id)
    if self.conf.ldap.group_members_are_ids:
        user_dn = user_ref['id']
    else:
        user_dn = user_ref['dn']
    return self.group.list_user_groups_filtered(user_dn, hints)