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
def update_flowspec_vrf_table(self, flowspec_family, route_dist, rules, actions=None, is_withdraw=False):
    """Update a BGP route in the VRF table for Flow Specification.

        ``flowspec_family`` specifies one of the flowspec family name.

        ``route_dist`` specifies a route distinguisher value.

        ``rules`` specifies NLRIs of Flow Specification as
        a dictionary type value.

        `` actions`` specifies Traffic Filtering Actions of
        Flow Specification as a dictionary type value.

        If `is_withdraw` is False, which is the default, add a BGP route
        to the VRF table identified by `route_dist`.
        If `is_withdraw` is True, remove a BGP route from the VRF table.
        """
    from os_ken.services.protocols.bgp.core import BgpCoreError
    from os_ken.services.protocols.bgp.api.prefix import FLOWSPEC_FAMILY_VPNV4, FLOWSPEC_FAMILY_VPNV6, FLOWSPEC_FAMILY_L2VPN
    if flowspec_family == FLOWSPEC_FAMILY_VPNV4:
        vrf_table = self._tables.get((route_dist, VRF_RF_IPV4_FLOWSPEC))
        prefix = FlowSpecIPv4NLRI.from_user(**rules)
        try:
            communities = create_v4flowspec_actions(actions)
        except ValueError as e:
            raise BgpCoreError(desc=str(e))
    elif flowspec_family == FLOWSPEC_FAMILY_VPNV6:
        vrf_table = self._tables.get((route_dist, VRF_RF_IPV6_FLOWSPEC))
        prefix = FlowSpecIPv6NLRI.from_user(**rules)
        try:
            communities = create_v6flowspec_actions(actions)
        except ValueError as e:
            raise BgpCoreError(desc=str(e))
    elif flowspec_family == FLOWSPEC_FAMILY_L2VPN:
        vrf_table = self._tables.get((route_dist, VRF_RF_L2VPN_FLOWSPEC))
        prefix = FlowSpecL2VPNNLRI.from_user(route_dist, **rules)
        try:
            communities = create_l2vpnflowspec_actions(actions)
        except ValueError as e:
            raise BgpCoreError(desc=str(e))
    else:
        raise BgpCoreError(desc='Unsupported flowspec_family %s' % flowspec_family)
    if vrf_table is None:
        raise BgpCoreError(desc='VRF table does not exist: route_dist=%s, flowspec_family=%s' % (route_dist, flowspec_family))
    vrf_table.insert_vrffs_path(nlri=prefix, communities=communities, is_withdraw=is_withdraw)