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
def recv_bfd_pkt(self, datapath, in_port, data):
    pkt = packet.Packet(data)
    eth = pkt.get_protocols(ethernet.ethernet)[0]
    if eth.ethertype != ETH_TYPE_IP:
        return
    ip_pkt = pkt.get_protocols(ipv4.ipv4)[0]
    if ip_pkt.ttl != 255:
        return
    bfd_pkt = BFDPacket.bfd_parse(data)
    if not isinstance(bfd_pkt, bfd.bfd):
        return
    if bfd_pkt.ver != 1:
        return
    if bfd_pkt.flags & bfd.BFD_FLAG_AUTH_PRESENT:
        if bfd_pkt.length < 26:
            return
    elif bfd_pkt.length < 24:
        return
    if bfd_pkt.detect_mult == 0:
        return
    if bfd_pkt.flags & bfd.BFD_FLAG_MULTIPOINT:
        return
    if bfd_pkt.my_discr == 0:
        return
    if bfd_pkt.your_discr != 0 and bfd_pkt.your_discr not in self.session:
        return
    if bfd_pkt.your_discr == 0 and bfd_pkt.state not in [bfd.BFD_STATE_ADMIN_DOWN, bfd.BFD_STATE_DOWN]:
        return
    sess_my_discr = None
    if bfd_pkt.your_discr == 0:
        for s in self.session.values():
            if s.dpid == datapath.id and s.ofport == in_port:
                sess_my_discr = s.my_discr
                break
        if sess_my_discr is None:
            return
    else:
        sess_my_discr = bfd_pkt.your_discr
    sess = self.session[sess_my_discr]
    if bfd_pkt.flags & bfd.BFD_FLAG_AUTH_PRESENT and sess._auth_type == 0:
        return
    if bfd_pkt.flags & bfd.BFD_FLAG_AUTH_PRESENT == 0 and sess._auth_type != 0:
        return
    if bfd_pkt.flags & bfd.BFD_FLAG_AUTH_PRESENT:
        if sess._auth_type == 0:
            return
        if bfd_pkt.auth_cls.auth_type != sess._auth_type:
            return
        if sess._auth_type in [bfd.BFD_AUTH_KEYED_MD5, bfd.BFD_AUTH_METICULOUS_KEYED_MD5, bfd.BFD_AUTH_KEYED_SHA1, bfd.BFD_AUTH_METICULOUS_KEYED_SHA1]:
            if sess._auth_seq_known:
                if bfd_pkt.auth_cls.seq < sess._rcv_auth_seq:
                    return
                if sess._auth_type in [bfd.BFD_AUTH_METICULOUS_KEYED_MD5, bfd.BFD_AUTH_METICULOUS_KEYED_SHA1]:
                    if bfd_pkt.auth_cls.seq <= sess._rcv_auth_seq:
                        return
                if bfd_pkt.auth_cls.seq > sess._rcv_auth_seq + 3 * sess._detect_mult:
                    return
        if not bfd_pkt.authenticate(sess._auth_keys):
            LOG.debug('[BFD][%s][AUTH] BFD Control authentication failed.', hex(sess._local_discr))
            return
    if sess is not None:
        if not sess._remote_addr_config:
            sess.set_remote_addr(eth.src, ip_pkt.src)
        sess.recv(bfd_pkt)