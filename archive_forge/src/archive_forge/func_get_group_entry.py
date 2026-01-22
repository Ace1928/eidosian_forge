from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def get_group_entry(self, uid):
    """
            get_group_entry returns an LDAP group entry for the given group UID by searching the internal cache
            of the LDAPInterface first, then sending an LDAP query if the cache did not contain the entry.
        """
    if uid in self.cached_groups:
        return (self.cached_groups.get(uid), None)
    group, err = self.groupQuery.ldap_search(self.connection, uid, self.required_group_attributes)
    if err:
        return (None, err)
    self.cached_groups[uid] = group
    return (group, None)