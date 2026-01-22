from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def enable_hostgroup_host_notifications(self, hostgroup):
    """
        Enables notifications for all hosts in a particular hostgroup.

        Note that this command does not enable notifications for
        services associated with the hosts in this hostgroup.

        Syntax: ENABLE_HOSTGROUP_HOST_NOTIFICATIONS;<hostgroup_name>
        """
    cmd = 'ENABLE_HOSTGROUP_HOST_NOTIFICATIONS'
    notif_str = self._fmt_notif_str(cmd, hostgroup)
    nagios_return = self._write_command(notif_str)
    if nagios_return:
        return notif_str
    else:
        return 'Fail: could not write to the command file'