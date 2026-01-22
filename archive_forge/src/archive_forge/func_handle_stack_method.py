import functools
from webob import exc
from heat.common.i18n import _
from heat.common import identifier
@functools.wraps(handler)
def handle_stack_method(controller, req, stack_name, stack_id, **kwargs):
    stack_identity = identifier.HeatIdentifier(req.context.tenant_id, stack_name, stack_id)
    return handler(controller, req, dict(stack_identity), **kwargs)