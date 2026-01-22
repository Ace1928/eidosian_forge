import logging
import netaddr
from os_ken.services.protocols.bgp.base import SUPPORTED_GLOBAL_RF
from os_ken.services.protocols.bgp.model import OutgoingRoute
from os_ken.services.protocols.bgp.peer import Peer
from os_ken.lib.packet.bgp import BGPPathAttributeCommunities
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_COMMUNITIES
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import RouteTargetMembershipNLRI
from os_ken.services.protocols.bgp.utils.bgp \
def req_rr_to_non_rtc_peers(self, route_family):
    """Makes refresh request to all peers for given address family.

        Skips making request to peer that have valid RTC capability.
        """
    assert route_family != RF_RTC_UC
    for peer in self._peers.values():
        if peer.in_established and peer.is_mbgp_cap_valid(route_family) and (not peer.is_mbgp_cap_valid(RF_RTC_UC)):
            peer.request_route_refresh(route_family)