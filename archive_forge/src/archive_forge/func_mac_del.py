import logging
from os_ken.lib.mac import haddr_to_str
def mac_del(self, dpid, mac):
    del self.mac_to_port[dpid][mac]