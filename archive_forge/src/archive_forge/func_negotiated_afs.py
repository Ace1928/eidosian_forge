import logging
import socket
import struct
import traceback
from socket import IPPROTO_TCP, TCP_NODELAY
from eventlet import semaphore
from os_ken.lib.packet import bgp
from os_ken.lib.packet.bgp import AS_TRANS
from os_ken.lib.packet.bgp import BGPMessage
from os_ken.lib.packet.bgp import BGPOpen
from os_ken.lib.packet.bgp import BGPUpdate
from os_ken.lib.packet.bgp import BGPKeepAlive
from os_ken.lib.packet.bgp import BGPNotification
from os_ken.lib.packet.bgp import BGP_MSG_OPEN
from os_ken.lib.packet.bgp import BGP_MSG_UPDATE
from os_ken.lib.packet.bgp import BGP_MSG_KEEPALIVE
from os_ken.lib.packet.bgp import BGP_MSG_NOTIFICATION
from os_ken.lib.packet.bgp import BGP_MSG_ROUTE_REFRESH
from os_ken.lib.packet.bgp import BGP_CAP_FOUR_OCTET_AS_NUMBER
from os_ken.lib.packet.bgp import BGP_CAP_ENHANCED_ROUTE_REFRESH
from os_ken.lib.packet.bgp import BGP_CAP_MULTIPROTOCOL
from os_ken.lib.packet.bgp import BGP_ERROR_HOLD_TIMER_EXPIRED
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_HOLD_TIMER_EXPIRED
from os_ken.lib.packet.bgp import get_rf
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import CORE_ERROR_CODE
from os_ken.services.protocols.bgp.constants import BGP_FSM_CONNECT
from os_ken.services.protocols.bgp.constants import BGP_FSM_OPEN_CONFIRM
from os_ken.services.protocols.bgp.constants import BGP_FSM_OPEN_SENT
from os_ken.services.protocols.bgp.constants import BGP_VERSION_NUM
from os_ken.services.protocols.bgp.protocol import Protocol
@property
def negotiated_afs(self):
    local_caps = self.sent_open_msg.opt_param
    remote_caps = self.recv_open_msg.opt_param
    local_mbgp_cap = [cap for cap in local_caps if cap.cap_code == BGP_CAP_MULTIPROTOCOL]
    remote_mbgp_cap = [cap for cap in remote_caps if cap.cap_code == BGP_CAP_MULTIPROTOCOL]
    if local_mbgp_cap and remote_mbgp_cap:
        local_families = set([(peer_cap.afi, peer_cap.safi) for peer_cap in local_mbgp_cap])
        remote_families = set([(peer_cap.afi, peer_cap.safi) for peer_cap in remote_mbgp_cap])
        afi_safi = local_families.intersection(remote_families)
    else:
        afi_safi = set()
    afs = []
    for afi, safi in afi_safi:
        afs.append(get_rf(afi, safi))
    return afs