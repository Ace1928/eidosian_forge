from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def validate_ldap_query(qry, isDNOnly=False):
    scope = qry.get('scope')
    if scope and scope not in ('', 'sub', 'one', 'base'):
        return 'invalid scope %s' % scope
    derefAlias = qry.get('derefAliases')
    if derefAlias and derefAlias not in ('never', 'search', 'base', 'always'):
        return ('not a valid LDAP alias dereferncing behavior: %s', derefAlias)
    timeout = qry.get('timeout')
    if timeout and float(timeout) < 0:
        return 'timeout must be equal to or greater than zero'
    qry_filter = qry.get('filter', '')
    if isDNOnly:
        if len(qry_filter) > 0:
            return 'cannot specify a filter when using "dn" as the UID attribute'
    elif len(qry_filter) == 0 or qry_filter[0] != '(':
        return "filter does not start with an '('"
    return None