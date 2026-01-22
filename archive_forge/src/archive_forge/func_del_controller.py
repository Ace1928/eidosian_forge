import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def del_controller(self):
    """
        Deletes the configured OpenFlow controller address.

        This method is corresponding to the following ovs-vsctl command::

            $ ovs-vsctl del-controller <bridge>
        """
    command = ovs_vsctl.VSCtlCommand('del-controller', [self.br_name])
    self.run_command([command])