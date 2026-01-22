import logging
import netaddr
import socket
from os_ken.lib.packet.bgp import BGP_ERROR_CEASE
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_CONNECTION_RESET
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_CONNECTION_COLLISION_RESOLUTION
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_INCOMPLETE
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import CORE_ERROR_CODE
from os_ken.services.protocols.bgp.constants import STD_BGP_SERVER_PORT_NUM
from os_ken.services.protocols.bgp import core_managers
from os_ken.services.protocols.bgp.model import FlexinetOutgoingRoute
from os_ken.services.protocols.bgp.protocol import Factory
from os_ken.services.protocols.bgp.signals.emit import BgpSignalBus
from os_ken.services.protocols.bgp.speaker import BgpProtocol
from os_ken.services.protocols.bgp.utils.rtfilter import RouteTargetManager
from os_ken.services.protocols.bgp.rtconf.neighbors import CONNECT_MODE_ACTIVE
from os_ken.services.protocols.bgp.utils import stats
from os_ken.services.protocols.bgp.bmp import BMPClient
from os_ken.lib import sockopt
from os_ken.lib import ip
def on_vrf_removed(self, route_dist):
    vrf_stats_timer = self._timers.get(route_dist)
    if vrf_stats_timer:
        vrf_stats_timer.stop()
        del self._timers[route_dist]