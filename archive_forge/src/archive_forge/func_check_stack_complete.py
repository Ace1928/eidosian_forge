import eventlet.queue
import functools
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common import exception
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import sync_point
from heat.objects import resource as resource_objects
from heat.rpc import api as rpc_api
from heat.rpc import listener_client
def check_stack_complete(cnxt, stack, current_traversal, sender_id, deps, is_update):
    """Mark the stack complete if the update is complete.

    Complete is currently in the sense that all desired resources are in
    service, not that superfluous ones have been cleaned up.
    """
    roots = set(deps.roots())
    if (sender_id, is_update) not in roots:
        return

    def mark_complete(stack_id, data):
        stack.mark_complete()
    sender_key = parser.ConvergenceNode(sender_id, is_update)
    sync_point.sync(cnxt, stack.id, current_traversal, True, mark_complete, roots, {sender_key: None})