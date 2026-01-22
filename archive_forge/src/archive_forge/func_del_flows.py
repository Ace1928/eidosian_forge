import binascii
import inspect
import json
import logging
import math
import netaddr
import os
import signal
import sys
import time
import traceback
from random import randint
from os_ken import cfg
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import hub
from os_ken.lib import stringify
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_5
def del_flows(self, cookie=0):
    """
        Delete all flow except default flow by using the cookie value.

        Note: In OpenFlow 1.0, DELETE and DELETE_STRICT commands can
        not be filtered by the cookie value and this value is ignored.
        """
    ofp = self.dp.ofproto
    parser = self.dp.ofproto_parser
    cookie_mask = 0
    if cookie:
        cookie_mask = 18446744073709551615
    if ofp.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
        match = parser.OFPMatch()
        mod = parser.OFPFlowMod(self.dp, match, cookie, ofp.OFPFC_DELETE)
    else:
        mod = parser.OFPFlowMod(self.dp, cookie=cookie, cookie_mask=cookie_mask, table_id=ofp.OFPTT_ALL, command=ofp.OFPFC_DELETE, out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY)
    return self.send_msg(mod)