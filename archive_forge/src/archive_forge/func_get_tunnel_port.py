import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def get_tunnel_port(self, name, tunnel_type='gre'):
    type_ = self.db_get_val('Interface', name, 'type')
    if type_ != tunnel_type:
        return
    options = self.db_get_map('Interface', name, 'options')
    if 'local_ip' in options and 'remote_ip' in options:
        ofport = self.get_ofport(name)
        return TunnelPort(name, ofport, tunnel_type, options['local_ip'], options['remote_ip'])