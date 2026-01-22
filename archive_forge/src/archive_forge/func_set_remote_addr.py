import logging
import time
import random
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.ofproto.ether import ETH_TYPE_IP, ETH_TYPE_ARP
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import inet
from os_ken.lib import hub
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import udp
from os_ken.lib.packet import bfd
from os_ken.lib.packet import arp
from os_ken.lib.packet.arp import ARP_REQUEST, ARP_REPLY
def set_remote_addr(self, dst_mac, dst_ip):
    """
        Configure remote ethernet and IP addresses.
        """
    self.dst_mac = dst_mac
    self.dst_ip = dst_ip
    if not (dst_mac == 'FF:FF:FF:FF:FF:FF' or dst_ip == '255.255.255.255'):
        self._remote_addr_config = True
    LOG.info('[BFD][%s][REMOTE] Remote address configured: %s, %s.', hex(self._local_discr), self.dst_ip, self.dst_mac)