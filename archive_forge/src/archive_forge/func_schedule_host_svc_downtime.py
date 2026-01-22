from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def schedule_host_svc_downtime(self, host, minutes=30, start=None):
    """
        This command is used to schedule downtime for
        all services associated with a particular host.

        During the specified downtime, Nagios will not send
        notifications out about the host.

        SCHEDULE_HOST_SVC_DOWNTIME;<host_name>;<start_time>;<end_time>;
        <fixed>;<trigger_id>;<duration>;<author>;<comment>
        """
    cmd = 'SCHEDULE_HOST_SVC_DOWNTIME'
    dt_cmd_str = self._fmt_dt_str(cmd, host, minutes, start=start)
    self._write_command(dt_cmd_str)