from os_ken.lib import hub
from os_ken.services.protocols.bgp.api.base import register
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
from os_ken.services.protocols.bgp.rtconf.common import CommonConf
@register(name='core.reset_neighbor')
def reset_neighbor(ip_address):
    neighs_conf = CORE_MANAGER.neighbors_conf
    neigh_conf = neighs_conf.get_neighbor_conf(ip_address)
    if not neigh_conf:
        raise RuntimeConfigError('No neighbor configuration found for given IP: %s' % ip_address)
    if neigh_conf.enabled:
        neigh_conf.enabled = False

        def up():
            neigh_conf.enabled = True
        hub.spawn_after(NEIGHBOR_RESET_WAIT_TIME, up)
    else:
        raise RuntimeConfigError('Neighbor %s is not enabled, hence cannot reset.' % ip_address)
    return True