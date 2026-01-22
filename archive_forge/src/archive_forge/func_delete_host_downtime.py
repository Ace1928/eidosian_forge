from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def delete_host_downtime(self, host, services=None, comment=None):
    """
        This command is used to remove scheduled downtime for a particular
        host.

        Syntax: DEL_DOWNTIME_BY_HOST_NAME;<host_name>;
        [<service_desription>];[<start_time>];[<comment>]
        """
    cmd = 'DEL_DOWNTIME_BY_HOST_NAME'
    if services is None:
        dt_del_cmd_str = self._fmt_dt_del_str(cmd, host, comment=comment)
        self._write_command(dt_del_cmd_str)
    else:
        for service in services:
            dt_del_cmd_str = self._fmt_dt_del_str(cmd, host, svc=service, comment=comment)
            self._write_command(dt_del_cmd_str)