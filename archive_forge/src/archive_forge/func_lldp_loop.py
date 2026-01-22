import logging
import struct
import time
from os_ken import cfg
from collections import defaultdict
from os_ken.topology import event
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from os_ken.exception import OSKenException
from os_ken.lib import addrconv, hub
from os_ken.lib.mac import DONTCARE_STR
from os_ken.lib.dpid import dpid_to_str, str_to_dpid
from os_ken.lib.port_no import port_no_to_str
from os_ken.lib.packet import packet, ethernet
from os_ken.lib.packet import lldp, ether_types
from os_ken.ofproto.ether import ETH_TYPE_LLDP
from os_ken.ofproto.ether import ETH_TYPE_CFM
from os_ken.ofproto import nx_match
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
def lldp_loop(self):
    while self.is_active:
        self.lldp_event.clear()
        now = time.time()
        timeout = None
        ports_now = []
        ports = []
        for key, data in self.ports.items():
            if data.timestamp is None:
                ports_now.append(key)
                continue
            expire = data.timestamp + self.LLDP_SEND_PERIOD_PER_PORT
            if expire <= now:
                ports.append(key)
                continue
            timeout = expire - now
            break
        for port in ports_now:
            self.send_lldp_packet(port)
        for port in ports:
            self.send_lldp_packet(port)
            hub.sleep(self.LLDP_SEND_GUARD)
        if timeout is not None and ports:
            timeout = 0
        self.lldp_event.wait(timeout=timeout)