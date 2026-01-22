from __future__ import (absolute_import, division, print_function)
import random
import time
from datetime import datetime, timedelta, timezone
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_list, check_type_str
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def check_boot_time(self, distribution, previous_boot_time):
    display.vvv('{action}: attempting to get system boot time'.format(action=self._task.action))
    connect_timeout = self._task.args.get('connect_timeout', self._task.args.get('connect_timeout_sec', self.DEFAULT_CONNECT_TIMEOUT))
    if connect_timeout:
        try:
            display.debug('{action}: setting connect_timeout to {value}'.format(action=self._task.action, value=connect_timeout))
            self._connection.set_option('connection_timeout', connect_timeout)
            self._connection.reset()
        except AttributeError:
            display.warning('Connection plugin does not allow the connection timeout to be overridden')
    try:
        current_boot_time = self.get_system_boot_time(distribution)
    except Exception as e:
        raise e
    if len(current_boot_time) == 0 or current_boot_time == previous_boot_time:
        raise ValueError('boot time has not changed')