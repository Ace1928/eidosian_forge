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
@register(name='vrf.update')
def update_vrf(**kwargs):
    route_dist = kwargs.get(ROUTE_DISTINGUISHER)
    vrf_id = kwargs.get(ConfWithId.ID)
    vrf_rf = kwargs.get(VRF_RF)
    vrf_conf = CORE_MANAGER.vrfs_conf.get_vrf_conf(route_dist, vrf_rf, vrf_id=vrf_id)
    if not vrf_conf:
        create_vrf(**kwargs)
    else:
        vrf_conf.update(**kwargs)
    return True