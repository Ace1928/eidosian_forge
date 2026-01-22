import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_2_parser import *
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ether
from os_ken.ofproto.ofproto_parser import MsgBase
from os_ken import utils
from os_ken.lib import addrconv
from os_ken.lib import pack_utils
def test_data_is_not_total_len(self):
    xid = 3423224276
    buffer_id = 2926809324
    reason = 128
    table_id = 3
    data = b'PacketIn'
    total_len = len(data) - 1
    self._test_parser(xid, buffer_id, total_len, reason, table_id, data)