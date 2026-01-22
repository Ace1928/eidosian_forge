import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def get_network_safe(self, dpid, port_no):
    port = self.get(dpid, {}).get(port_no)
    if port is None:
        return self.nw_id_unknown
    return port.network_id