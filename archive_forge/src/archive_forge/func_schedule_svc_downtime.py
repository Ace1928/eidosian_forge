from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def schedule_svc_downtime(self, host, services=None, minutes=30, start=None):
    """
        This command is used to schedule downtime for a particular
        service.

        During the specified downtime, Nagios will not send
        notifications out about the service.

        Syntax: SCHEDULE_SVC_DOWNTIME;<host_name>;<service_description>
        <start_time>;<end_time>;<fixed>;<trigger_id>;<duration>;<author>;
        <comment>
        """
    cmd = 'SCHEDULE_SVC_DOWNTIME'
    if services is None:
        services = []
    for service in services:
        dt_cmd_str = self._fmt_dt_str(cmd, host, minutes, start=start, svc=service)
        self._write_command(dt_cmd_str)