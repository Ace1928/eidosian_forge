from heat.common.i18n import _
from heat.engine.resources.openstack.heat import none_resource
from heat.engine import support
A resource for managing LBaaS v2 L7Policies.

    This resource manages Neutron-LBaaS v2 L7Policies, which represent
    a collection of L7Rules. L7Policy holds the action that should be performed
    when the rules are matched (Redirect to Pool, Redirect to URL, Reject).
    L7Policy holds a Listener id, so a Listener can evaluate a collection of
    L7Policies. L7Policy will return True when all of the L7Rules that belong
    to this L7Policy are matched. L7Policies under a specific Listener are
    ordered and the first l7Policy that returns a match will be executed.
    When none of the policies match the request gets forwarded to
    listener.default_pool_id.
    