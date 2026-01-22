from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def schedule_forced_host_check(self, host):
    """
        This command schedules a forced active check for a particular host.

        Syntax: SCHEDULE_FORCED_HOST_CHECK;<host_name>;<check_time>
        """
    cmd = 'SCHEDULE_FORCED_HOST_CHECK'
    chk_cmd_str = self._fmt_chk_str(cmd, host, svc=None)
    self._write_command(chk_cmd_str)