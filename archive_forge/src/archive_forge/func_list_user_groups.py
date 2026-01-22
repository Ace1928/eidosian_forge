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
def list_user_groups(self, user_dn):
    """Return a list of groups for which the user is a member."""
    user_dn_esc = ldap.filter.escape_filter_chars(user_dn)
    if self.group_ad_nesting:
        query = '(%s:%s:=%s)' % (self.member_attribute, LDAP_MATCHING_RULE_IN_CHAIN, user_dn_esc)
    else:
        query = '(%s=%s)' % (self.member_attribute, user_dn_esc)
    return self.get_all(query)