from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
A resource to create rule for some label.

    Resource for allowing specified label to measure the traffic for a specific
    set of ip range.
    