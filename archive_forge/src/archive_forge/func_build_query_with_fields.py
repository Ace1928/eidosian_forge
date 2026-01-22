from __future__ import (absolute_import, division, print_function)
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def build_query_with_fields(query, fields):
    """ for GET requests"""
    if fields is not None and query is None:
        query = {}
    if fields is not None:
        query['fields'] = fields
    return query