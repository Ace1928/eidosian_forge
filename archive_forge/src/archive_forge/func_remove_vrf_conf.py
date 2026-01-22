import abc
import logging
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_IPv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.services.protocols.bgp.utils import validation
from os_ken.services.protocols.bgp.base import get_validator
from os_ken.services.protocols.bgp.rtconf.base import BaseConf
from os_ken.services.protocols.bgp.rtconf.base import BaseConfListener
from os_ken.services.protocols.bgp.rtconf.base import ConfigTypeError
from os_ken.services.protocols.bgp.rtconf.base import ConfigValueError
from os_ken.services.protocols.bgp.rtconf.base import ConfWithId
from os_ken.services.protocols.bgp.rtconf.base import ConfWithIdListener
from os_ken.services.protocols.bgp.rtconf.base import ConfWithStats
from os_ken.services.protocols.bgp.rtconf.base import ConfWithStatsListener
from os_ken.services.protocols.bgp.rtconf.base import MAX_NUM_EXPORT_RT
from os_ken.services.protocols.bgp.rtconf.base import MAX_NUM_IMPORT_RT
from os_ken.services.protocols.bgp.rtconf.base import MULTI_EXIT_DISC
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
from os_ken.services.protocols.bgp.rtconf.base import SITE_OF_ORIGINS
from os_ken.services.protocols.bgp.rtconf.base import validate
from os_ken.services.protocols.bgp.rtconf.base import validate_med
from os_ken.services.protocols.bgp.rtconf.base import validate_soo_list
def remove_vrf_conf(self, route_dist=None, vrf_id=None, vrf_rf=None):
    """Removes any matching `VrfConf` for given `route_dist` or `vrf_id`

        Parameters:
            - `route_dist`: (str) route distinguisher of a configured VRF
            - `vrf_id`: (str) vrf ID
            - `vrf_rf`: (str) route family of the VRF configuration
        If only `route_dist` is given, removes `VrfConf`s for all supported
        address families for this `route_dist`. If `vrf_rf` is given, than only
        removes `VrfConf` for that specific route family. If only `vrf_id` is
        given, matching `VrfConf` will be removed.
        """
    if route_dist is None and vrf_id is None:
        raise RuntimeConfigError(desc='To delete supply route_dist or id.')
    vrf_rfs = SUPPORTED_VRF_RF
    if vrf_rf:
        vrf_rfs = vrf_rf
    removed_vrf_confs = []
    for route_family in vrf_rfs:
        if route_dist is not None:
            rd_rf_id = VrfConf.create_rd_rf_id(route_dist, route_family)
            vrf_conf = self._vrfs_by_rd_rf.pop(rd_rf_id, None)
            if vrf_conf:
                self._vrfs_by_id.pop(vrf_conf.id, None)
                removed_vrf_confs.append(vrf_conf)
        else:
            vrf_conf = self._vrfs_by_id.pop(vrf_id, None)
            if vrf_conf:
                self._vrfs_by_rd_rf.pop(vrf_conf.rd_rd_id, None)
                removed_vrf_confs.append(vrf_conf)
    for vrf_conf in removed_vrf_confs:
        self._notify_listeners(VrfsConf.REMOVE_VRF_CONF_EVT, vrf_conf)
    return removed_vrf_confs