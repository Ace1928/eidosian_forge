from os_ken.services.protocols.bgp.rtconf.base import ConfWithStats
from os_ken.services.protocols.bgp.rtconf.common import CommonConfListener
from os_ken.services.protocols.bgp.rtconf.neighbors import NeighborsConfListener
from os_ken.services.protocols.bgp.rtconf import vrfs
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfConf
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfsConfListener
import logging
def on_add_vrf_conf(self, evt):
    """Event handler for new VrfConf.

        Creates a VrfTable to store routing information related to new Vrf.
        Also arranges for related paths to be imported to this VrfTable.
        """
    vrf_conf = evt.value
    route_family = vrf_conf.route_family
    assert route_family in vrfs.SUPPORTED_VRF_RF
    vrf_table = self._table_manager.create_and_link_vrf_table(vrf_conf)
    vrf_conf.add_listener(ConfWithStats.UPDATE_STATS_LOG_ENABLED_EVT, self.on_stats_config_change)
    vrf_conf.add_listener(ConfWithStats.UPDATE_STATS_TIME_EVT, self.on_stats_config_change)
    vrf_conf.add_listener(VrfConf.VRF_CHG_EVT, self.on_chg_vrf_conf)
    self._table_manager.import_all_vpn_paths_to_vrf(vrf_table)
    self._rt_manager.update_local_rt_nlris()
    self._signal_bus.vrf_added(vrf_conf)