from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def openshift_ldap_build_base_query(config):
    qry = {}
    if config.get('baseDN'):
        qry['base'] = config.get('baseDN')
    scope = determine_ldap_scope(config.get('scope'))
    if scope:
        qry['scope'] = scope
    pageSize = config.get('pageSize')
    if pageSize and int(pageSize) > 0:
        qry['sizelimit'] = int(pageSize)
    timeout = config.get('timeout')
    if timeout and int(timeout) > 0:
        qry['timeout'] = int(timeout)
    filter = config.get('filter')
    if filter:
        qry['filterstr'] = filter
    derefAlias = determine_deref_aliases(config.get('derefAliases'))
    if derefAlias:
        qry['derefAlias'] = derefAlias
    return qry