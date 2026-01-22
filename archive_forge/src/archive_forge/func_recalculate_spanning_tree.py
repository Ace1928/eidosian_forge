import datetime
import logging
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.exception import OFPUnknownVersion
from os_ken.lib import hub
from os_ken.lib import mac
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib.packet import bpdu
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import llc
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
def recalculate_spanning_tree(self, init=True):
    """ Re-calculation of spanning tree. """
    for port in self.ports.values():
        if port.state is not PORT_STATE_DISABLE:
            port.down(PORT_STATE_BLOCK, msg_init=init)
    if init:
        self.send_event(EventTopologyChange(self.dp))
    port_roles = {}
    self.root_priority = Priority(self.bridge_id, 0, None, None)
    self.root_times = self.bridge_times
    if init:
        self.logger.info('Root bridge.', extra=self.dpid_str)
        for port_no in self.ports:
            port_roles[port_no] = DESIGNATED_PORT
    else:
        port_roles, self.root_priority, self.root_times = self._spanning_tree_algorithm()
    for port_no, role in port_roles.items():
        if self.ports[port_no].state is not PORT_STATE_DISABLE:
            self.ports[port_no].up(role, self.root_priority, self.root_times)