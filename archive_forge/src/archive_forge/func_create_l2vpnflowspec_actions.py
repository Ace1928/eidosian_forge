import logging
import netaddr
from os_ken.lib import ip
from os_ken.lib.packet.bgp import (
from os_ken.services.protocols.bgp.info_base.rtc import RtcPath
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Path
from os_ken.services.protocols.bgp.info_base.ipv6 import Ipv6Path
from os_ken.services.protocols.bgp.info_base.vpnv4 import Vpnv4Path
from os_ken.services.protocols.bgp.info_base.vpnv6 import Vpnv6Path
from os_ken.services.protocols.bgp.info_base.evpn import EvpnPath
from os_ken.services.protocols.bgp.info_base.ipv4fs import IPv4FlowSpecPath
from os_ken.services.protocols.bgp.info_base.ipv6fs import IPv6FlowSpecPath
from os_ken.services.protocols.bgp.info_base.vpnv4fs import VPNv4FlowSpecPath
from os_ken.services.protocols.bgp.info_base.vpnv6fs import VPNv6FlowSpecPath
from os_ken.services.protocols.bgp.info_base.l2vpnfs import L2VPNFlowSpecPath
def create_l2vpnflowspec_actions(actions=None):
    """
    Create list of traffic filtering actions for L2VPN Flow Specification.
    """
    from os_ken.services.protocols.bgp.api.prefix import FLOWSPEC_ACTION_TRAFFIC_RATE, FLOWSPEC_ACTION_TRAFFIC_ACTION, FLOWSPEC_ACTION_REDIRECT, FLOWSPEC_ACTION_TRAFFIC_MARKING, FLOWSPEC_ACTION_VLAN, FLOWSPEC_ACTION_TPID
    action_types = {FLOWSPEC_ACTION_TRAFFIC_RATE: BGPFlowSpecTrafficRateCommunity, FLOWSPEC_ACTION_TRAFFIC_ACTION: BGPFlowSpecTrafficActionCommunity, FLOWSPEC_ACTION_REDIRECT: BGPFlowSpecRedirectCommunity, FLOWSPEC_ACTION_TRAFFIC_MARKING: BGPFlowSpecTrafficMarkingCommunity, FLOWSPEC_ACTION_VLAN: BGPFlowSpecVlanActionCommunity, FLOWSPEC_ACTION_TPID: BGPFlowSpecTPIDActionCommunity}
    return _create_actions(actions, action_types)