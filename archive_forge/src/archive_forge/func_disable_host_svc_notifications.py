from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def disable_host_svc_notifications(self, host):
    """
        This command is used to prevent notifications from being sent
        out for all services on the specified host.

        Note that this command does not disable notifications from
        being sent out about the host.

        Syntax: DISABLE_HOST_SVC_NOTIFICATIONS;<host_name>
        """
    cmd = 'DISABLE_HOST_SVC_NOTIFICATIONS'
    notif_str = self._fmt_notif_str(cmd, host)
    self._write_command(notif_str)