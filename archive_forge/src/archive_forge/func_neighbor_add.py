import netaddr
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.packet.bgp import (
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
from os_ken.services.protocols.bgp.signals.emit import BgpSignalBus
from os_ken.services.protocols.bgp.api.base import call
from os_ken.services.protocols.bgp.api.base import PREFIX
from os_ken.services.protocols.bgp.api.base import EVPN_ROUTE_TYPE
from os_ken.services.protocols.bgp.api.base import EVPN_ESI
from os_ken.services.protocols.bgp.api.base import EVPN_ETHERNET_TAG_ID
from os_ken.services.protocols.bgp.api.base import REDUNDANCY_MODE
from os_ken.services.protocols.bgp.api.base import IP_ADDR
from os_ken.services.protocols.bgp.api.base import MAC_ADDR
from os_ken.services.protocols.bgp.api.base import NEXT_HOP
from os_ken.services.protocols.bgp.api.base import IP_PREFIX
from os_ken.services.protocols.bgp.api.base import GW_IP_ADDR
from os_ken.services.protocols.bgp.api.base import ROUTE_DISTINGUISHER
from os_ken.services.protocols.bgp.api.base import ROUTE_FAMILY
from os_ken.services.protocols.bgp.api.base import EVPN_VNI
from os_ken.services.protocols.bgp.api.base import TUNNEL_TYPE
from os_ken.services.protocols.bgp.api.base import PMSI_TUNNEL_TYPE
from os_ken.services.protocols.bgp.api.base import MAC_MOBILITY
from os_ken.services.protocols.bgp.api.base import TUNNEL_ENDPOINT_IP
from os_ken.services.protocols.bgp.api.prefix import EVPN_MAX_ET
from os_ken.services.protocols.bgp.api.prefix import ESI_TYPE_LACP
from os_ken.services.protocols.bgp.api.prefix import ESI_TYPE_L2_BRIDGE
from os_ken.services.protocols.bgp.api.prefix import ESI_TYPE_MAC_BASED
from os_ken.services.protocols.bgp.api.prefix import EVPN_ETH_AUTO_DISCOVERY
from os_ken.services.protocols.bgp.api.prefix import EVPN_MAC_IP_ADV_ROUTE
from os_ken.services.protocols.bgp.api.prefix import EVPN_MULTICAST_ETAG_ROUTE
from os_ken.services.protocols.bgp.api.prefix import EVPN_ETH_SEGMENT
from os_ken.services.protocols.bgp.api.prefix import EVPN_IP_PREFIX_ROUTE
from os_ken.services.protocols.bgp.api.prefix import REDUNDANCY_MODE_ALL_ACTIVE
from os_ken.services.protocols.bgp.api.prefix import REDUNDANCY_MODE_SINGLE_ACTIVE
from os_ken.services.protocols.bgp.api.prefix import TUNNEL_TYPE_VXLAN
from os_ken.services.protocols.bgp.api.prefix import TUNNEL_TYPE_NVGRE
from os_ken.services.protocols.bgp.api.prefix import (
from os_ken.services.protocols.bgp.api.prefix import (
from os_ken.services.protocols.bgp.model import ReceivedRoute
from os_ken.services.protocols.bgp.rtconf.common import LOCAL_AS
from os_ken.services.protocols.bgp.rtconf.common import ROUTER_ID
from os_ken.services.protocols.bgp.rtconf.common import CLUSTER_ID
from os_ken.services.protocols.bgp.rtconf.common import BGP_SERVER_HOSTS
from os_ken.services.protocols.bgp.rtconf.common import BGP_SERVER_PORT
from os_ken.services.protocols.bgp.rtconf.common import DEFAULT_BGP_SERVER_HOSTS
from os_ken.services.protocols.bgp.rtconf.common import DEFAULT_BGP_SERVER_PORT
from os_ken.services.protocols.bgp.rtconf.common import (
from os_ken.services.protocols.bgp.rtconf.common import DEFAULT_LABEL_RANGE
from os_ken.services.protocols.bgp.rtconf.common import REFRESH_MAX_EOR_TIME
from os_ken.services.protocols.bgp.rtconf.common import REFRESH_STALEPATH_TIME
from os_ken.services.protocols.bgp.rtconf.common import LABEL_RANGE
from os_ken.services.protocols.bgp.rtconf.common import ALLOW_LOCAL_AS_IN_COUNT
from os_ken.services.protocols.bgp.rtconf.common import LOCAL_PREF
from os_ken.services.protocols.bgp.rtconf.common import DEFAULT_LOCAL_PREF
from os_ken.services.protocols.bgp.rtconf import neighbors
from os_ken.services.protocols.bgp.rtconf import vrfs
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_IPV4
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_IPV6
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_VPNV4
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_VPNV6
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_EVPN
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_IPV4FS
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_IPV6FS
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_VPNV4FS
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_VPNV6FS
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_L2VPNFS
from os_ken.services.protocols.bgp.rtconf.base import CAP_ENHANCED_REFRESH
from os_ken.services.protocols.bgp.rtconf.base import CAP_FOUR_OCTET_AS_NUMBER
from os_ken.services.protocols.bgp.rtconf.base import MULTI_EXIT_DISC
from os_ken.services.protocols.bgp.rtconf.base import SITE_OF_ORIGINS
from os_ken.services.protocols.bgp.rtconf.neighbors import (
from os_ken.services.protocols.bgp.rtconf.vrfs import SUPPORTED_VRF_RF
from os_ken.services.protocols.bgp.info_base.base import Filter
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Path
from os_ken.services.protocols.bgp.info_base.ipv6 import Ipv6Path
from os_ken.services.protocols.bgp.info_base.vpnv4 import Vpnv4Path
from os_ken.services.protocols.bgp.info_base.vpnv6 import Vpnv6Path
from os_ken.services.protocols.bgp.info_base.evpn import EvpnPath
def neighbor_add(self, address, remote_as, remote_port=DEFAULT_BGP_PORT, enable_ipv4=DEFAULT_CAP_MBGP_IPV4, enable_ipv6=DEFAULT_CAP_MBGP_IPV6, enable_vpnv4=DEFAULT_CAP_MBGP_VPNV4, enable_vpnv6=DEFAULT_CAP_MBGP_VPNV6, enable_evpn=DEFAULT_CAP_MBGP_EVPN, enable_ipv4fs=DEFAULT_CAP_MBGP_IPV4FS, enable_ipv6fs=DEFAULT_CAP_MBGP_IPV6FS, enable_vpnv4fs=DEFAULT_CAP_MBGP_VPNV4FS, enable_vpnv6fs=DEFAULT_CAP_MBGP_VPNV6FS, enable_l2vpnfs=DEFAULT_CAP_MBGP_L2VPNFS, enable_enhanced_refresh=DEFAULT_CAP_ENHANCED_REFRESH, enable_four_octet_as_number=DEFAULT_CAP_FOUR_OCTET_AS_NUMBER, next_hop=None, password=None, multi_exit_disc=None, site_of_origins=None, is_route_server_client=DEFAULT_IS_ROUTE_SERVER_CLIENT, is_route_reflector_client=DEFAULT_IS_ROUTE_REFLECTOR_CLIENT, is_next_hop_self=DEFAULT_IS_NEXT_HOP_SELF, local_address=None, local_port=None, local_as=None, connect_mode=DEFAULT_CONNECT_MODE):
    """ This method registers a new neighbor. The BGP speaker tries to
        establish a bgp session with the peer (accepts a connection
        from the peer and also tries to connect to it).

        ``address`` specifies the IP address of the peer. It must be
        the string representation of an IP address. Only IPv4 is
        supported now.

        ``remote_as`` specifies the AS number of the peer. It must be
        an integer between 1 and 65535.

        ``remote_port`` specifies the TCP port number of the peer.

        ``enable_ipv4`` enables IPv4 address family for this
        neighbor.

        ``enable_ipv6`` enables IPv6 address family for this
        neighbor.

        ``enable_vpnv4`` enables VPNv4 address family for this
        neighbor.

        ``enable_vpnv6`` enables VPNv6 address family for this
        neighbor.

        ``enable_evpn`` enables Ethernet VPN address family for this
        neighbor.

        ``enable_ipv4fs`` enables IPv4 Flow Specification address family
        for this neighbor.

        ``enable_ipv6fs`` enables IPv6 Flow Specification address family
        for this neighbor.

        ``enable_vpnv4fs`` enables VPNv4 Flow Specification address family
        for this neighbor.

        ``enable_vpnv6fs`` enables VPNv6 Flow Specification address family
        for this neighbor.

        ``enable_l2vpnfs`` enables L2VPN Flow Specification address family
        for this neighbor.

        ``enable_enhanced_refresh`` enables Enhanced Route Refresh for this
        neighbor.

        ``enable_four_octet_as_number`` enables Four-Octet AS Number
        capability for this neighbor.

        ``next_hop`` specifies the next hop IP address. If not
        specified, host's ip address to access to a peer is used.

        ``password`` is used for the MD5 authentication if it's
        specified. By default, the MD5 authentication is disabled.

        ``multi_exit_disc`` specifies multi exit discriminator (MED) value
        as an int type value.
        If omitted, MED is not sent to the neighbor.

        ``site_of_origins`` specifies site_of_origin values.
        This parameter must be a list of string.

        ``is_route_server_client`` specifies whether this neighbor is a
        router server's client or not.

        ``is_route_reflector_client`` specifies whether this neighbor is a
        router reflector's client or not.

        ``is_next_hop_self`` specifies whether the BGP speaker announces
        its own ip address to iBGP neighbor or not as path's next_hop address.

        ``local_address`` specifies Loopback interface address for
        iBGP peering.

        ``local_port`` specifies source TCP port for iBGP peering.

        ``local_as`` specifies local AS number per-peer.
        If omitted, the AS number of BGPSpeaker instance is used.

        ``connect_mode`` specifies how to connect to this neighbor.
        This parameter must be one of the following.

        - CONNECT_MODE_ACTIVE         = 'active'
        - CONNECT_MODE_PASSIVE        = 'passive'
        - CONNECT_MODE_BOTH (default) = 'both'
        """
    bgp_neighbor = {neighbors.IP_ADDRESS: address, neighbors.REMOTE_AS: remote_as, REMOTE_PORT: remote_port, PEER_NEXT_HOP: next_hop, PASSWORD: password, IS_ROUTE_SERVER_CLIENT: is_route_server_client, IS_ROUTE_REFLECTOR_CLIENT: is_route_reflector_client, IS_NEXT_HOP_SELF: is_next_hop_self, CONNECT_MODE: connect_mode, CAP_ENHANCED_REFRESH: enable_enhanced_refresh, CAP_FOUR_OCTET_AS_NUMBER: enable_four_octet_as_number, CAP_MBGP_IPV4: enable_ipv4, CAP_MBGP_IPV6: enable_ipv6, CAP_MBGP_VPNV4: enable_vpnv4, CAP_MBGP_VPNV6: enable_vpnv6, CAP_MBGP_EVPN: enable_evpn, CAP_MBGP_IPV4FS: enable_ipv4fs, CAP_MBGP_IPV6FS: enable_ipv6fs, CAP_MBGP_VPNV4FS: enable_vpnv4fs, CAP_MBGP_VPNV6FS: enable_vpnv6fs, CAP_MBGP_L2VPNFS: enable_l2vpnfs}
    if multi_exit_disc:
        bgp_neighbor[MULTI_EXIT_DISC] = multi_exit_disc
    if site_of_origins:
        bgp_neighbor[SITE_OF_ORIGINS] = site_of_origins
    if local_address:
        bgp_neighbor[LOCAL_ADDRESS] = local_address
    if local_port:
        bgp_neighbor[LOCAL_PORT] = local_port
    if local_as:
        bgp_neighbor[LOCAL_AS] = local_as
    call('neighbor.create', **bgp_neighbor)