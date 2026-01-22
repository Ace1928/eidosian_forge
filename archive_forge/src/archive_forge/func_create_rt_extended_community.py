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
def create_rt_extended_community(value, subtype=2):
    """
    Creates an instance of the BGP Route Target Community (if "subtype=2")
    or Route Origin Community ("subtype=3").

    :param value: String of Route Target or Route Origin value.
    :param subtype: Subtype of Extended Community.
    :return: An instance of Route Target or Route Origin Community.
    """
    global_admin, local_admin = value.split(':')
    local_admin = int(local_admin)
    if global_admin.isdigit() and 0 <= int(global_admin) <= 65535:
        ext_com = BGPTwoOctetAsSpecificExtendedCommunity(subtype=subtype, as_number=int(global_admin), local_administrator=local_admin)
    elif global_admin.isdigit() and 65535 < int(global_admin) <= 4294967295:
        ext_com = BGPFourOctetAsSpecificExtendedCommunity(subtype=subtype, as_number=int(global_admin), local_administrator=local_admin)
    elif ip.valid_ipv4(global_admin):
        ext_com = BGPIPv4AddressSpecificExtendedCommunity(subtype=subtype, ipv4_address=global_admin, local_administrator=local_admin)
    else:
        raise ValueError('Invalid Route Target or Route Origin value: %s' % value)
    return ext_com