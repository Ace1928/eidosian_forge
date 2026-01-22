from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
A resource for managing Neutron L2 Gateway Connections.

    The L2 Gateway Connection provides a mapping to connect a Neutron network
    to a L2 Gateway on a particular segmentation ID.
    