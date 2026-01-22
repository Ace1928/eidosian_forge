import random
import re
import shelve
import ldap
from oslo_log import log
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common
def delete_ext_s(self, dn, serverctrls, clientctrls=None):
    """Remove the ldap object at specified dn."""
    if server_fail:
        raise ldap.SERVER_DOWN
    try:
        children = self._getChildren(dn)
        if children:
            raise ldap.NOT_ALLOWED_ON_NONLEAF
    except KeyError:
        LOG.debug('delete item failed: dn=%s not found.', dn)
        raise ldap.NO_SUCH_OBJECT
    super(FakeLdapNoSubtreeDelete, self).delete_ext_s(dn, serverctrls, clientctrls)