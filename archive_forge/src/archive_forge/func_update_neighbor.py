import logging
from os_ken.services.protocols.bgp.api.base import register
from os_ken.services.protocols.bgp.api.base import RegisterWithArgChecks
from os_ken.services.protocols.bgp.api.base import FLOWSPEC_FAMILY
from os_ken.services.protocols.bgp.api.base import FLOWSPEC_RULES
from os_ken.services.protocols.bgp.api.base import FLOWSPEC_ACTIONS
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
from os_ken.services.protocols.bgp.rtconf.base import ConfWithId
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
from os_ken.services.protocols.bgp.rtconf import neighbors
from os_ken.services.protocols.bgp.rtconf.neighbors import NeighborConf
from os_ken.services.protocols.bgp.rtconf.vrfs import ROUTE_DISTINGUISHER
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV4
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfConf
from os_ken.services.protocols.bgp import constants as const
@RegisterWithArgChecks(name='neighbor.update', req_args=[neighbors.IP_ADDRESS, neighbors.CHANGES])
def update_neighbor(neigh_ip_address, changes):
    rets = []
    for k, v in changes.items():
        if k == neighbors.MULTI_EXIT_DISC:
            rets.append(_update_med(neigh_ip_address, v))
        if k == neighbors.ENABLED:
            rets.append(update_neighbor_enabled(neigh_ip_address, v))
        if k == neighbors.CONNECT_MODE:
            rets.append(_update_connect_mode(neigh_ip_address, v))
    return all(rets)