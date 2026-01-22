import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def db_get_map(self, table, record, column):
    """
        Gets dict type value of 'column' in 'record' in 'table'.

        This method is corresponding to the following ovs-vsctl command::

            $ ovs-vsctl get TBL REC COL
        """
    val = self.db_get_val(table, record, column)
    assert isinstance(val, dict)
    return val