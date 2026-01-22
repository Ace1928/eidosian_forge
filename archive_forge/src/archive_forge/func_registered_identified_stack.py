import functools
from webob import exc
from heat.common.i18n import _
from heat.common import identifier
def registered_identified_stack(handler):
    """Decorator that passes a stack identifier instead of path components.

    This is a handler method decorator. Policy is enforced using a registered
    policy name.
    """
    return registered_policy_enforce(_identified_stack(handler))