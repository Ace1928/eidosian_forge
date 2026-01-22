from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def disable_servicegroup_svc_notifications(self, servicegroup):
    """
        This command is used to prevent notifications from being sent
        out for all services in the specified servicegroup.

        Note that this does not prevent notifications from being sent
        out about the hosts in this servicegroup.

        Syntax: DISABLE_SERVICEGROUP_SVC_NOTIFICATIONS;<servicegroup_name>
        """
    cmd = 'DISABLE_SERVICEGROUP_SVC_NOTIFICATIONS'
    notif_str = self._fmt_notif_str(cmd, servicegroup)
    self._write_command(notif_str)