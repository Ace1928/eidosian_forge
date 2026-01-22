from __future__ import (absolute_import, division, print_function)
import random
import time
from datetime import datetime, timedelta, timezone
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_list, check_type_str
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def perform_reboot(self, task_vars, distribution):
    result = {}
    reboot_result = {}
    shutdown_command = self.get_shutdown_command(task_vars, distribution)
    shutdown_command_args = self.get_shutdown_command_args(distribution)
    reboot_command = '{0} {1}'.format(shutdown_command, shutdown_command_args)
    try:
        display.vvv('{action}: rebooting server...'.format(action=self._task.action))
        display.debug("{action}: rebooting server with command '{command}'".format(action=self._task.action, command=reboot_command))
        reboot_result = self._low_level_execute_command(reboot_command, sudoable=self.DEFAULT_SUDOABLE)
    except AnsibleConnectionFailure as e:
        display.debug('{action}: AnsibleConnectionFailure caught and handled: {error}'.format(action=self._task.action, error=to_text(e)))
        reboot_result['rc'] = 0
    result['start'] = datetime.now(timezone.utc)
    if reboot_result['rc'] != 0:
        result['failed'] = True
        result['rebooted'] = False
        result['msg'] = "Reboot command failed. Error was: '{stdout}, {stderr}'".format(stdout=to_native(reboot_result['stdout'].strip()), stderr=to_native(reboot_result['stderr'].strip()))
        return result
    result['failed'] = False
    return result