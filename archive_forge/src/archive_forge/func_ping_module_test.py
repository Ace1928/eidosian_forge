from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime, timedelta, timezone
from ansible.module_utils.common.text.converters import to_text
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def ping_module_test(connect_timeout):
    """ Test ping module, if available """
    display.vvv('wait_for_connection: attempting ping module test')
    if self._discovered_interpreter_key:
        task_vars['ansible_facts'].pop(self._discovered_interpreter_key, None)
    try:
        self._connection.reset()
    except AttributeError:
        pass
    ping_result = self._execute_module(module_name='ansible.legacy.ping', module_args=dict(), task_vars=task_vars)
    if ping_result['ping'] != 'pong':
        raise Exception('ping test failed')