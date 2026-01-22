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
@RegisterWithArgChecks(name='neighbor.attribute_map.set', req_args=[neighbors.IP_ADDRESS, neighbors.ATTRIBUTE_MAP], opt_args=[ROUTE_DISTINGUISHER, VRF_RF])
def set_neighbor_attribute_map(neigh_ip_address, at_maps, route_dist=None, route_family=VRF_RF_IPV4):
    """set attribute_maps to the neighbor."""
    core = CORE_MANAGER.get_core_service()
    peer = core.peer_manager.get_by_addr(neigh_ip_address)
    at_maps_key = const.ATTR_MAPS_LABEL_DEFAULT
    at_maps_dict = {}
    if route_dist is not None:
        vrf_conf = CORE_MANAGER.vrfs_conf.get_vrf_conf(route_dist, route_family)
        if vrf_conf:
            at_maps_key = ':'.join([route_dist, route_family])
        else:
            raise RuntimeConfigError(desc='No VrfConf with rd %s' % route_dist)
    at_maps_dict[const.ATTR_MAPS_LABEL_KEY] = at_maps_key
    at_maps_dict[const.ATTR_MAPS_VALUE] = at_maps
    peer.attribute_maps = at_maps_dict
    return True