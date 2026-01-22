import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def port_added(self, datapath, port_no):
    if port_no == 0 or port_no >= datapath.ofproto.OFPP_MAX:
        return
    self.dpids.setdefault_network(datapath.id, port_no)