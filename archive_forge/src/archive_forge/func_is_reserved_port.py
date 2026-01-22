import contextlib
import logging
import random
from socket import IPPROTO_TCP
from socket import TCP_NODELAY
from socket import SHUT_WR
from socket import timeout as SocketTimeout
import ssl
from os_ken import cfg
from os_ken.lib import hub
from os_ken.lib.hub import StreamServer
import os_ken.base.app_manager
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import nx_match
from os_ken.controller import ofp_event
from os_ken.controller.handler import HANDSHAKE_DISPATCHER, DEAD_DISPATCHER
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib import ip
def is_reserved_port(self, port_no):
    return port_no > self.ofproto.OFPP_MAX