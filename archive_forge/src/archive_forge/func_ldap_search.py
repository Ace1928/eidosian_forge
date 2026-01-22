from __future__ import (absolute_import, division, print_function)
import re
def ldap_search(filter, base=None, attr=None):
    """Replaces uldaps search and uses a generator.
    !! Arguments are not the same."""
    if base is None:
        base = base_dn()
    msgid = uldap().lo.lo.search(base, ldap_module().SCOPE_SUBTREE, filterstr=filter, attrlist=attr)
    while True:
        result_type, result_data = uldap().lo.lo.result(msgid, all=0)
        if not result_data:
            break
        if result_type is ldap_module().RES_SEARCH_RESULT:
            break
        elif result_type is ldap_module().RES_SEARCH_ENTRY:
            for res in result_data:
                yield res
    uldap().lo.lo.abandon(msgid)