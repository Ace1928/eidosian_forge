from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def schedule_forced_svc_check(self, host, services=None):
    """
        This command schedules a forced active check for a particular
        service.

        Syntax: SCHEDULE_FORCED_SVC_CHECK;<host_name>;<service_description>;<check_time>
        """
    cmd = 'SCHEDULE_FORCED_SVC_CHECK'
    if services is None:
        services = []
    for service in services:
        chk_cmd_str = self._fmt_chk_str(cmd, host, svc=service)
        self._write_command(chk_cmd_str)