import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def get_tunnel_ports(self, tunnel_type='gre'):
    get_tunnel_port = functools.partial(self.get_tunnel_port, tunnel_type=tunnel_type)
    return self._get_ports(get_tunnel_port)