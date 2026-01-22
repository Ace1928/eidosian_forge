import copy
from heat.common import exception
from heat.common.i18n import _
from heat.engine import scheduler
Notify the LoadBalancer to reload its config.

    This must be done after activation (instance in ACTIVE state), otherwise
    the instances' IP addresses may not be available.
    