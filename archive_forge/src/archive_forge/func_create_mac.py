import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def create_mac(self, network_id, dpid, port_no, mac_address):
    self.mac_addresses.add_port(network_id, dpid, port_no, mac_address)
    self.dpids.set_mac(network_id, dpid, port_no, mac_address)