from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def enable_host_notifications(self, host):
    """
        Enables notifications for a particular host.

        Note that this command does not enable notifications for
        services associated with this host.

        Syntax: ENABLE_HOST_NOTIFICATIONS;<host_name>
        """
    cmd = 'ENABLE_HOST_NOTIFICATIONS'
    notif_str = self._fmt_notif_str(cmd, host)
    self._write_command(notif_str)