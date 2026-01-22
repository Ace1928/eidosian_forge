import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_4 as ofproto
@staticmethod
def register_flow_update_event(event, length):

    def _register_flow_update_event(cls):
        OFPFlowUpdateHeader._EVENT[event] = cls
        cls.cls_flow_update_event = event
        cls.cls_flow_update_length = length
        return cls
    return _register_flow_update_event