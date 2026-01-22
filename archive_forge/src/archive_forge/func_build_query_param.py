import collections
import os
from urllib import parse
import uuid
import stevedore
from cinderclient import exceptions
def build_query_param(params, sort=False):
    """parse list to url query parameters"""
    if not params:
        return ''
    if not sort:
        param_list = list(params.items())
    else:
        param_list = list(sorted(params.items()))
    query_string = parse.urlencode([(k, v) for k, v in param_list if v not in (None, '')])
    query_string = query_string.replace('%7E=', '~=')
    if query_string:
        query_string = '?%s' % (query_string,)
    return query_string