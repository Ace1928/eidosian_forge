import logging
from os_ken.lib.mac import haddr_to_str
def mac_list(self, dpid, port):
    return [mac for mac, port_ in self.mac_to_port.get(dpid, {}).items() if port_ == port]