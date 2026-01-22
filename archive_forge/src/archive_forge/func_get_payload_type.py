import struct
import logging
from os_ken.lib import stringify
from . import packet_base
from . import packet_utils
from . import bgp
from . import openflow
from . import zebra
@staticmethod
def get_payload_type(src_port, dst_port):
    from os_ken.ofproto.ofproto_common import OFP_TCP_PORT, OFP_SSL_PORT_OLD
    if bgp.TCP_SERVER_PORT in [src_port, dst_port]:
        return bgp.BGPMessage
    elif src_port in [OFP_TCP_PORT, OFP_SSL_PORT_OLD] or dst_port in [OFP_TCP_PORT, OFP_SSL_PORT_OLD]:
        return openflow.openflow
    elif src_port == zebra.ZEBRA_PORT:
        return zebra._ZebraMessageFromZebra
    elif dst_port == zebra.ZEBRA_PORT:
        return zebra.ZebraMessage
    else:
        return None