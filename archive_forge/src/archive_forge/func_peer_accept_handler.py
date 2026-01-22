from os_ken import cfg
import socket
import netaddr
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
from os_ken.lib import rpc
from os_ken.lib import hub
from os_ken.lib import mac
def peer_accept_handler(self, new_sock, addr):
    peer = Peer(self._rpc_events)
    table = {rpc.MessageType.REQUEST: peer._handle_vrrp_request}
    peer._endpoint = rpc.EndPoint(new_sock, disp_table=table)
    self._peers.append(peer)
    hub.spawn(self._peer_loop_thread, peer)