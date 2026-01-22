from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def enable_servicegroup_host_notifications(self, servicegroup):
    """
        Enables notifications for all hosts that have services that
        are members of a particular servicegroup.

        Note that this command does not enable notifications for
        services associated with the hosts in this servicegroup.

        Syntax: ENABLE_SERVICEGROUP_HOST_NOTIFICATIONS;<servicegroup_name>
        """
    cmd = 'ENABLE_SERVICEGROUP_HOST_NOTIFICATIONS'
    notif_str = self._fmt_notif_str(cmd, servicegroup)
    nagios_return = self._write_command(notif_str)
    if nagios_return:
        return notif_str
    else:
        return 'Fail: could not write to the command file'