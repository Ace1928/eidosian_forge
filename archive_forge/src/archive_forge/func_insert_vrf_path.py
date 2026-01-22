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
def insert_vrf_path(self, nlri, next_hop=None, gen_lbl=False, is_withdraw=False, **kwargs):
    assert nlri
    pattrs = None
    label_list = []
    vrf_conf = self.vrf_conf
    if not is_withdraw:
        table_manager = self._core_service.table_manager
        if gen_lbl and next_hop:
            label_key = (vrf_conf.route_dist, next_hop)
            nh_label = table_manager.get_nexthop_label(label_key)
            if not nh_label:
                nh_label = table_manager.get_next_vpnv4_label()
                table_manager.set_nexthop_label(label_key, nh_label)
            label_list.append(nh_label)
        elif gen_lbl:
            label_list.append(table_manager.get_next_vpnv4_label())
        if gen_lbl and isinstance(nlri, EvpnMacIPAdvertisementNLRI):
            nlri.mpls_labels = label_list[:2]
        elif gen_lbl and isinstance(nlri, EvpnIpPrefixNLRI):
            nlri.mpls_label = label_list[0]
        pattrs = OrderedDict()
        from os_ken.services.protocols.bgp.core import EXPECTED_ORIGIN
        pattrs[BGP_ATTR_TYPE_ORIGIN] = BGPPathAttributeOrigin(EXPECTED_ORIGIN)
        pattrs[BGP_ATTR_TYPE_AS_PATH] = BGPPathAttributeAsPath([])
        communities = []
        if isinstance(nlri, EvpnEthernetSegmentNLRI):
            subtype = 2
            es_import = nlri.esi.mac_addr
            communities.append(BGPEvpnEsImportRTExtendedCommunity(subtype=subtype, es_import=es_import))
        for rt in vrf_conf.export_rts:
            communities.append(create_rt_extended_community(rt, 2))
        for soo in vrf_conf.soo_list:
            communities.append(create_rt_extended_community(soo, 3))
        tunnel_type = kwargs.get('tunnel_type', None)
        if tunnel_type:
            communities.append(BGPEncapsulationExtendedCommunity.from_str(tunnel_type))
        redundancy_mode = kwargs.get('redundancy_mode', None)
        if redundancy_mode is not None:
            subtype = 1
            flags = 0
            from os_ken.services.protocols.bgp.api.prefix import REDUNDANCY_MODE_SINGLE_ACTIVE
            if redundancy_mode == REDUNDANCY_MODE_SINGLE_ACTIVE:
                flags |= BGPEvpnEsiLabelExtendedCommunity.SINGLE_ACTIVE_BIT
            vni = kwargs.get('vni', None)
            if vni is not None:
                communities.append(BGPEvpnEsiLabelExtendedCommunity(subtype=subtype, flags=flags, vni=vni))
            else:
                communities.append(BGPEvpnEsiLabelExtendedCommunity(subtype=subtype, flags=flags, mpls_label=label_list[0]))
        mac_mobility_seq = kwargs.get('mac_mobility', None)
        if mac_mobility_seq is not None:
            from os_ken.lib.packet.bgp import BGPEvpnMacMobilityExtendedCommunity
            is_static = mac_mobility_seq == -1
            communities.append(BGPEvpnMacMobilityExtendedCommunity(subtype=0, flags=1 if is_static else 0, sequence_number=mac_mobility_seq if not is_static else 0))
        pattrs[BGP_ATTR_TYPE_EXTENDED_COMMUNITIES] = BGPPathAttributeExtendedCommunities(communities=communities)
        if vrf_conf.multi_exit_disc:
            pattrs[BGP_ATTR_TYPE_MULTI_EXIT_DISC] = BGPPathAttributeMultiExitDisc(vrf_conf.multi_exit_disc)
        pmsi_tunnel_type = kwargs.get('pmsi_tunnel_type', None)
        if pmsi_tunnel_type is not None:
            from os_ken.services.protocols.bgp.api.prefix import PMSI_TYPE_INGRESS_REP
            if pmsi_tunnel_type == PMSI_TYPE_INGRESS_REP:
                vtep = kwargs.get('tunnel_endpoint_ip', self._core_service.router_id)
                tunnel_id = PmsiTunnelIdIngressReplication(tunnel_endpoint_ip=vtep)
            else:
                tunnel_id = None
            pattrs[BGP_ATTR_TYEP_PMSI_TUNNEL_ATTRIBUTE] = BGPPathAttributePmsiTunnel(pmsi_flags=0, tunnel_type=pmsi_tunnel_type, tunnel_id=tunnel_id, vni=kwargs.get('vni', None))
    puid = self.VRF_PATH_CLASS.create_puid(vrf_conf.route_dist, nlri.prefix)
    path = self.VRF_PATH_CLASS(puid, None, nlri, 0, pattrs=pattrs, nexthop=next_hop, label_list=label_list, is_withdraw=is_withdraw)
    eff_dest = self.insert(path)
    self._signal_bus.dest_changed(eff_dest)
    return label_list