import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
return a tuple which can be used as *args for
        ofproto_v1_0_parser.OFPMatch.__init__().
        see Datapath.send_flow_mod.
        