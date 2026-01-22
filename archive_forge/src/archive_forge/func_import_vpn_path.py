import abc
import logging
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
from os_ken.lib.packet.bgp import BGP_ATTR_TYEP_PMSI_TUNNEL_ATTRIBUTE
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import EvpnEthernetSegmentNLRI
from os_ken.lib.packet.bgp import BGPPathAttributeExtendedCommunities
from os_ken.lib.packet.bgp import BGPPathAttributeMultiExitDisc
from os_ken.lib.packet.bgp import BGPEncapsulationExtendedCommunity
from os_ken.lib.packet.bgp import BGPEvpnEsiLabelExtendedCommunity
from os_ken.lib.packet.bgp import BGPEvpnEsImportRTExtendedCommunity
from os_ken.lib.packet.bgp import BGPPathAttributePmsiTunnel
from os_ken.lib.packet.bgp import PmsiTunnelIdIngressReplication
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import EvpnMacIPAdvertisementNLRI
from os_ken.lib.packet.bgp import EvpnIpPrefixNLRI
from os_ken.lib.packet.safi import (
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.constants import VPN_TABLE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
from os_ken.services.protocols.bgp.utils.bgp import create_rt_extended_community
from os_ken.services.protocols.bgp.utils.stats import LOCAL_ROUTES
from os_ken.services.protocols.bgp.utils.stats import REMOTE_ROUTES
from os_ken.services.protocols.bgp.utils.stats import RESOURCE_ID
from os_ken.services.protocols.bgp.utils.stats import RESOURCE_NAME
def import_vpn_path(self, vpn_path):
    """Imports `vpnv(4|6)_path` into `vrf(4|6)_table` or `evpn_path`
        into vrfevpn_table`.

        :Parameters:
            - `vpn_path`: (Path) VPN path that will be cloned and imported
            into VRF.
        Note: Does not do any checking if this import is valid.
        """
    assert vpn_path.route_family == self.VPN_ROUTE_FAMILY
    source = vpn_path.source
    if not source:
        source = VRF_TABLE
    if self.VPN_ROUTE_FAMILY == RF_L2_EVPN:
        vrf_nlri = vpn_path.nlri
    elif self.ROUTE_FAMILY.safi in [IP_FLOWSPEC, VPN_FLOWSPEC]:
        vrf_nlri = self.NLRI_CLASS(rules=vpn_path.nlri.rules)
    else:
        ip, masklen = vpn_path.nlri.prefix.split('/')
        vrf_nlri = self.NLRI_CLASS(length=int(masklen), addr=ip)
    vrf_path = self.VRF_PATH_CLASS(puid=self.VRF_PATH_CLASS.create_puid(vpn_path.nlri.route_dist, vpn_path.nlri.prefix), source=source, nlri=vrf_nlri, src_ver_num=vpn_path.source_version_num, pattrs=vpn_path.pathattr_map, nexthop=vpn_path.nexthop, is_withdraw=vpn_path.is_withdraw, label_list=getattr(vpn_path.nlri, 'label_list', None))
    if self._is_vrf_path_already_in_table(vrf_path):
        return None
    if self._is_vrf_path_filtered_out_by_import_maps(vrf_path):
        return None
    else:
        vrf_dest = self.insert(vrf_path)
        self._signal_bus.dest_changed(vrf_dest)