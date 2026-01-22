import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def find_db_attributes(self, table, *conditions):
    """
        Lists records satisfying 'conditions' in 'table'.

        This method is corresponding to the following ovs-vsctl command::

            $ ovs-vsctl find TBL CONDITION...

        .. Note::

            Currently, only '=' condition is supported.
            To support other condition is TODO.
        """
    args = [table]
    args.extend(conditions)
    command = ovs_vsctl.VSCtlCommand('find', args)
    self.run_command([command])
    if command.result:
        return command.result
    return []