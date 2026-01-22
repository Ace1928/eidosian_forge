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
def send_port_stats(self):
    """ Get port stats."""
    ofp = self.dp.ofproto
    parser = self.dp.ofproto_parser
    flags = 0
    if ofp.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
        port = ofp.OFPP_NONE
    else:
        port = ofp.OFPP_ANY
    req = parser.OFPPortStatsRequest(self.dp, flags, port)
    return self.send_msg(req)