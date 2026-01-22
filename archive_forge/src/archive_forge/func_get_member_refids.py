from heat.common import exception
from heat.common.i18n import _
from heat.engine import status
from heat.engine import template
from heat.rpc import api as rpc_api
def get_member_refids(group):
    """Get a list of member resources managed by the specified group.

    The list of resources is sorted first by created_time then by name.
    """
    return [r.FnGetRefId() for r in get_members(group)]