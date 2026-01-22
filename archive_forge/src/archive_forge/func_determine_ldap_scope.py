from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def determine_ldap_scope(scope):
    if scope in ('', 'sub'):
        return ldap.SCOPE_SUBTREE
    elif scope == 'base':
        return ldap.SCOPE_BASE
    elif scope == 'one':
        return ldap.SCOPE_ONELEVEL
    return None