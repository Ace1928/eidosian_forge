import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def get_datapath_id(self):
    """
        Gets Datapath ID of OVS instance.

        This method is corresponding to the following ovs-vsctl command::

            $ ovs-vsctl get Bridge <bridge> datapath_id
        """
    return self.db_get_val('Bridge', self.br_name, 'datapath_id')