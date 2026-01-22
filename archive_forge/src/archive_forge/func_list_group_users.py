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
def list_group_users(self, group_id):
    """Return a list of user dns which are members of a group."""
    group_ref = self.get(group_id)
    group_dn = group_ref['dn']
    try:
        if self.group_ad_nesting:
            attrs = self._ldap_get_list(self.tree_dn, self.LDAP_SCOPE, query_params={'member:%s:' % LDAP_MATCHING_RULE_IN_CHAIN: group_dn}, attrlist=[self.member_attribute])
        else:
            attrs = self._ldap_get_list(group_dn, ldap.SCOPE_BASE, attrlist=[self.member_attribute])
    except ldap.NO_SUCH_OBJECT:
        raise self.NotFound(group_id=group_id)
    users = []
    for dn, member in attrs:
        user_dns = member.get(self.member_attribute, [])
        for user_dn in user_dns:
            users.append(user_dn)
    return users