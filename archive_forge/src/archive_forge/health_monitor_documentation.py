from heat.common.i18n import _
from heat.engine.resources.openstack.heat import none_resource
from heat.engine import support
A resource to handle load balancer health monitors.

    This resource creates and manages Neutron LBaaS v2 healthmonitors,
    which watches status of the load balanced servers.
    