from __future__ import (absolute_import, division, print_function)
import random
import time
from datetime import datetime, timedelta, timezone
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_list, check_type_str
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def get_system_boot_time(self, distribution):
    boot_time_command = self._get_value_from_facts('BOOT_TIME_COMMANDS', distribution, 'DEFAULT_BOOT_TIME_COMMAND')
    if self._task.args.get('boot_time_command'):
        boot_time_command = self._task.args.get('boot_time_command')
        try:
            check_type_str(boot_time_command, allow_conversion=False)
        except TypeError as e:
            raise AnsibleError("Invalid value given for 'boot_time_command': %s." % to_native(e))
    display.debug("{action}: getting boot time with command: '{command}'".format(action=self._task.action, command=boot_time_command))
    command_result = self._low_level_execute_command(boot_time_command, sudoable=self.DEFAULT_SUDOABLE)
    if command_result['rc'] != 0:
        stdout = command_result['stdout']
        stderr = command_result['stderr']
        raise AnsibleError('{action}: failed to get host boot time info, rc: {rc}, stdout: {out}, stderr: {err}'.format(action=self._task.action, rc=command_result['rc'], out=to_native(stdout), err=to_native(stderr)))
    display.debug('{action}: last boot time: {boot}'.format(action=self._task.action, boot=command_result['stdout'].strip()))
    return command_result['stdout'].strip()