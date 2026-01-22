import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def get_quantum_ports(self, port_name):
    LOG.debug('port_name %s', port_name)
    command = ovs_vsctl.VSCtlCommand('list-ifaces-verbose', [dpid_lib.dpid_to_str(self.datapath_id), port_name])
    self.run_command([command])
    if command.result:
        return command.result[0]
    return None