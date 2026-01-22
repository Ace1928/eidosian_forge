from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def silence_host(self, host):
    """
        This command is used to prevent notifications from being sent
        out for the host and all services on the specified host.

        This is equivalent to calling disable_host_svc_notifications
        and disable_host_notifications.

        Syntax: DISABLE_HOST_SVC_NOTIFICATIONS;<host_name>
        Syntax: DISABLE_HOST_NOTIFICATIONS;<host_name>
        """
    cmd = ['DISABLE_HOST_SVC_NOTIFICATIONS', 'DISABLE_HOST_NOTIFICATIONS']
    nagios_return = True
    return_str_list = []
    for c in cmd:
        notif_str = self._fmt_notif_str(c, host)
        nagios_return = self._write_command(notif_str) and nagios_return
        return_str_list.append(notif_str)
    if nagios_return:
        return return_str_list
    else:
        return 'Fail: could not write to the command file'