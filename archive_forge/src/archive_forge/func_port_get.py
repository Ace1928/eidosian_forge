import logging
from os_ken.lib.mac import haddr_to_str
def port_get(self, dpid, mac):
    return self.mac_to_port[dpid].get(mac)