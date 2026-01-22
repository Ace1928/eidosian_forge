from __future__ import (absolute_import, division, print_function)
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def build_query_with_timeout(query, timeout):
    """ for POST, PATCH, DELETE requests"""
    params = {} if query else None
    if timeout > 0:
        params = dict(return_timeout=timeout)
    if query is not None:
        params.update(query)
    return params