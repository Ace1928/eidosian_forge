import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def pipework(self, bridge, ip_addr, intf_name='', version=4):
    if not self.is_running:
        LOG.warning('Call run() before pipeworking')
        return
    c = CmdBuffer(' ')
    c << 'pipework {0}'.format(bridge.name)
    if intf_name != '':
        c << '-i {0}'.format(intf_name)
    else:
        intf_name = 'eth1'
    ipv4 = None
    ipv6 = None
    if version == 4:
        ipv4 = ip_addr
    else:
        c << '-a 6'
        ipv6 = ip_addr
    c << '{0} {1}'.format(self.docker_name(), ip_addr)
    self.set_addr_info(bridge=bridge.name, ipv4=ipv4, ipv6=ipv6, ifname=intf_name)
    self.execute(str(c), sudo=True, retry=True)