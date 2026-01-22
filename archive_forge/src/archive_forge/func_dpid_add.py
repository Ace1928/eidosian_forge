import logging
from os_ken.lib.mac import haddr_to_str
def dpid_add(self, dpid):
    LOG.debug('dpid_add: 0x%016x', dpid)
    self.mac_to_port.setdefault(dpid, {})