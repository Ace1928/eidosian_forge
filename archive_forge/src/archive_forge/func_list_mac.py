import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def list_mac(self, dpid, port_no):
    mac_address = self.dpids.get_mac(dpid, port_no)
    if mac_address is None:
        return []
    return [mac_address]