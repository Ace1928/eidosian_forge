import logging
from collections import OrderedDict
import netaddr
from os_ken.services.protocols.bgp.base import SUPPORTED_GLOBAL_RF
from os_ken.services.protocols.bgp.info_base.rtc import RtcTable
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Path
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Table
from os_ken.services.protocols.bgp.info_base.ipv6 import Ipv6Path
from os_ken.services.protocols.bgp.info_base.ipv6 import Ipv6Table
from os_ken.services.protocols.bgp.info_base.vpnv4 import Vpnv4Table
from os_ken.services.protocols.bgp.info_base.vpnv6 import Vpnv6Table
from os_ken.services.protocols.bgp.info_base.vrf4 import Vrf4Table
from os_ken.services.protocols.bgp.info_base.vrf6 import Vrf6Table
from os_ken.services.protocols.bgp.info_base.vrfevpn import VrfEvpnTable
from os_ken.services.protocols.bgp.info_base.evpn import EvpnTable
from os_ken.services.protocols.bgp.info_base.ipv4fs import IPv4FlowSpecPath
from os_ken.services.protocols.bgp.info_base.ipv4fs import IPv4FlowSpecTable
from os_ken.services.protocols.bgp.info_base.vpnv4fs import VPNv4FlowSpecTable
from os_ken.services.protocols.bgp.info_base.vrf4fs import Vrf4FlowSpecTable
from os_ken.services.protocols.bgp.info_base.ipv6fs import IPv6FlowSpecPath
from os_ken.services.protocols.bgp.info_base.ipv6fs import IPv6FlowSpecTable
from os_ken.services.protocols.bgp.info_base.vpnv6fs import VPNv6FlowSpecTable
from os_ken.services.protocols.bgp.info_base.vrf6fs import Vrf6FlowSpecTable
from os_ken.services.protocols.bgp.info_base.l2vpnfs import L2VPNFlowSpecTable
from os_ken.services.protocols.bgp.info_base.vrfl2vpnfs import L2vpnFlowSpecPath
from os_ken.services.protocols.bgp.info_base.vrfl2vpnfs import L2vpnFlowSpecTable
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV4
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV6
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_L2_EVPN
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV4_FLOWSPEC
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV6_FLOWSPEC
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_L2VPN_FLOWSPEC
from os_ken.services.protocols.bgp.rtconf.vrfs import SUPPORTED_VRF_RF
from os_ken.services.protocols.bgp.utils.bgp import create_v4flowspec_actions
from os_ken.services.protocols.bgp.utils.bgp import create_v6flowspec_actions
from os_ken.services.protocols.bgp.utils.bgp import create_l2vpnflowspec_actions
from os_ken.lib import type_desc
from os_ken.lib import ip
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_IPv4_VPN
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_IPv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGPPathAttributeExtendedCommunities
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_IGP
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
from os_ken.lib.packet.bgp import EvpnEsi
from os_ken.lib.packet.bgp import EvpnArbitraryEsi
from os_ken.lib.packet.bgp import EvpnNLRI
from os_ken.lib.packet.bgp import EvpnMacIPAdvertisementNLRI
from os_ken.lib.packet.bgp import EvpnInclusiveMulticastEthernetTagNLRI
from os_ken.lib.packet.bgp import IPAddrPrefix
from os_ken.lib.packet.bgp import IP6AddrPrefix
from os_ken.lib.packet.bgp import FlowSpecIPv4NLRI
from os_ken.lib.packet.bgp import FlowSpecIPv6NLRI
from os_ken.lib.packet.bgp import FlowSpecL2VPNNLRI
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv4
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv4_prefix
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv6
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv6_prefix
def update_vrf_table(self, route_dist, prefix=None, next_hop=None, route_family=None, route_type=None, tunnel_type=None, is_withdraw=False, redundancy_mode=None, pmsi_tunnel_type=None, tunnel_endpoint_ip=None, mac_mobility=None, **kwargs):
    """Update a BGP route in the VRF table identified by `route_dist`
        with the given `next_hop`.

        If `is_withdraw` is False, which is the default, add a BGP route
        to the VRF table identified by `route_dist` with the given
        `next_hop`.
        If `is_withdraw` is True, remove a BGP route from the VRF table
        and the given `next_hop` is ignored.

        If `route_family` is VRF_RF_L2_EVPN, `route_type` and `kwargs`
        are required to construct EVPN NLRI and `prefix` is ignored.

        ``redundancy_mode`` specifies a redundancy mode type.

`       `pmsi_tunnel_type` specifies the type of the PMSI tunnel attribute
         used to encode the multicast tunnel identifier.
        This field is advertised only if route_type is
        EVPN_MULTICAST_ETAG_ROUTE.

        `tunnel_endpoint_ip` specifies a tunnel endpoint IP other than the
        default local router ID; only used in EVPN_MULTICAST_ETAG_ROUTE

        `mac_mobility` specifies an optional integer sequence number to insert
        as a MAC Mobility extended community; special value `-1` is used for
        static MACs (MAC Mobility sequence 0 with STATIC flag set)

        Returns assigned VPN label.
        """
    from os_ken.services.protocols.bgp.core import BgpCoreError
    assert route_dist
    if is_withdraw:
        gen_lbl = False
        next_hop = None
    else:
        gen_lbl = True
        if not (is_valid_ipv4(next_hop) or is_valid_ipv6(next_hop)):
            raise BgpCoreError(desc='Invalid IPv4/IPv6 nexthop: %s' % next_hop)
    vrf_table = self._tables.get((route_dist, route_family))
    if vrf_table is None:
        raise BgpCoreError(desc='VRF table  does not exist: route_dist=%s, route_family=%s' % (route_dist, route_family))
    vni = kwargs.get('vni', None)
    if route_family == VRF_RF_IPV4:
        if not is_valid_ipv4_prefix(prefix):
            raise BgpCoreError(desc='Invalid IPv4 prefix: %s' % prefix)
        ip, masklen = prefix.split('/')
        prefix = IPAddrPrefix(int(masklen), ip)
    elif route_family == VRF_RF_IPV6:
        if not is_valid_ipv6_prefix(prefix):
            raise BgpCoreError(desc='Invalid IPv6 prefix: %s' % prefix)
        ip6, masklen = prefix.split('/')
        prefix = IP6AddrPrefix(int(masklen), ip6)
    elif route_family == VRF_RF_L2_EVPN:
        assert route_type
        if route_type == EvpnMacIPAdvertisementNLRI.ROUTE_TYPE_NAME:
            kwargs['mpls_labels'] = []
        if route_type == EvpnInclusiveMulticastEthernetTagNLRI.ROUTE_TYPE_NAME:
            vni = kwargs.pop('vni', None)
        subclass = EvpnNLRI._lookup_type_name(route_type)
        kwargs['route_dist'] = route_dist
        esi = kwargs.get('esi', None)
        if esi is not None:
            if isinstance(esi, dict):
                esi_type = esi.get('type', 0)
                esi_class = EvpnEsi._lookup_type(esi_type)
                kwargs['esi'] = esi_class.from_jsondict(esi)
            else:
                kwargs['esi'] = EvpnArbitraryEsi(type_desc.Int9.from_user(esi))
        if vni is not None:
            from os_ken.services.protocols.bgp.api.prefix import TUNNEL_TYPE_VXLAN, TUNNEL_TYPE_NVGRE
            assert tunnel_type in [None, TUNNEL_TYPE_VXLAN, TUNNEL_TYPE_NVGRE]
            gen_lbl = False
        prefix = subclass(**kwargs)
    else:
        raise BgpCoreError(desc='Unsupported route family %s' % route_family)
    return vrf_table.insert_vrf_path(nlri=prefix, next_hop=next_hop, gen_lbl=gen_lbl, is_withdraw=is_withdraw, redundancy_mode=redundancy_mode, mac_mobility=mac_mobility, vni=vni, tunnel_type=tunnel_type, pmsi_tunnel_type=pmsi_tunnel_type)