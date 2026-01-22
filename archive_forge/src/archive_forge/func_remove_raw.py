import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def remove_raw(self, network_id, dpid, port_no):
    ports = self[network_id]
    if (dpid, port_no) in ports:
        ports.remove((dpid, port_no))
        self._remove_event(network_id, dpid, port_no)