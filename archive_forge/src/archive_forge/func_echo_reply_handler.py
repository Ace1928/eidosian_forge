import itertools
import logging
import warnings
import os_ken.base.app_manager
from os_ken.lib import hub
from os_ken import utils
from os_ken.controller import ofp_event
from os_ken.controller.controller import OpenFlowController
from os_ken.controller.handler import set_ev_handler
from os_ken.controller.handler import HANDSHAKE_DISPATCHER, CONFIG_DISPATCHER,\
from os_ken.ofproto import ofproto_parser
@set_ev_handler(ofp_event.EventOFPEchoReply, [HANDSHAKE_DISPATCHER, CONFIG_DISPATCHER, MAIN_DISPATCHER])
def echo_reply_handler(self, ev):
    msg = ev.msg
    datapath = msg.datapath
    datapath.acknowledge_echo_reply(msg.xid)