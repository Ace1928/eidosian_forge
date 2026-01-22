from heat.common import exception
from heat.common.i18n import _
from heat.engine import status
from heat.engine import template
from heat.rpc import api as rpc_api
def get_nested_attrs(stack, key, use_indices, *path):
    path = key.split('.', 2)[1:] + list(path)
    if len(path) > 1:
        return get_rsrc_attr(stack, key, use_indices, *path)
    else:
        return get_rsrc_id(stack, key, use_indices, *path)