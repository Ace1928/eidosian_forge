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
def get_unknown_opttrans_attr(path):
    """Utility method that gives a `dict` of unknown and unsupported optional
    transitive path attributes of `path`.

    Returns dict: <key> - attribute type code, <value> - unknown path-attr.
    """
    path_attrs = path.pathattr_map
    unknown_opt_tran_attrs = {}
    for _, attr in path_attrs.items():
        if isinstance(attr, BGPPathAttributeUnknown) and attr.flags & (BGP_ATTR_FLAG_OPTIONAL | BGP_ATTR_FLAG_TRANSITIVE) or isinstance(attr, BGPPathAttributeAs4Path) or isinstance(attr, BGPPathAttributeAs4Aggregator):
            unknown_opt_tran_attrs[attr.type] = attr
    return unknown_opt_tran_attrs