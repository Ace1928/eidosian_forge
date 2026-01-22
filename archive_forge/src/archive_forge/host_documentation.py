from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
A resource to manage Blazar hosts.

    Host resource manages the physical hosts for the lease/reservation
    within OpenStack.

    # TODO(asmita): Based on an agreement with Blazar team, this resource
    class does not support updating host resource as currently Blazar does
    not support to delete existing extra_capability keys while updating host.
    Also, in near future, when Blazar team will come up with a new alternative
    API to resolve this issue, we will need to modify this class.
    