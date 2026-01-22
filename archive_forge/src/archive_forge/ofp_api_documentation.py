import netaddr
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.lib import ip
from . import ofp_event

    Unregister the given switch address.

    Unregisters the given switch address to let
    os_ken.controller.controller.OpenFlowController stop trying to initiate
    connection to switch.

    :param addr: A tuple of (host, port) pair of switch.
    