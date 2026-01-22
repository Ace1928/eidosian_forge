from __future__ import (absolute_import, division, print_function)
import random
import time
from datetime import datetime, timedelta, timezone
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_list, check_type_str
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def do_until_success_or_timeout(self, action, reboot_timeout, action_desc, distribution, action_kwargs=None):
    max_end_time = datetime.now(timezone.utc) + timedelta(seconds=reboot_timeout)
    if action_kwargs is None:
        action_kwargs = {}
    fail_count = 0
    max_fail_sleep = 12
    last_error_msg = ''
    while datetime.now(timezone.utc) < max_end_time:
        try:
            action(distribution=distribution, **action_kwargs)
            if action_desc:
                display.debug('{action}: {desc} success'.format(action=self._task.action, desc=action_desc))
            return
        except Exception as e:
            if isinstance(e, AnsibleConnectionFailure):
                try:
                    self._connection.reset()
                except AnsibleConnectionFailure:
                    pass
            random_int = random.randint(0, 1000) / 1000
            fail_sleep = 2 ** fail_count + random_int
            if fail_sleep > max_fail_sleep:
                fail_sleep = max_fail_sleep + random_int
            if action_desc:
                try:
                    error = to_text(e).splitlines()[-1]
                except IndexError as e:
                    error = to_text(e)
                last_error_msg = f"{self._task.action}: {action_desc} fail '{error}'"
                msg = f'{last_error_msg}, retrying in {fail_sleep:.4f} seconds...'
                display.debug(msg)
                display.vvv(msg)
            fail_count += 1
            time.sleep(fail_sleep)
    if last_error_msg:
        msg = f'Last error message before the timeout exception - {last_error_msg}'
        display.debug(msg)
        display.vvv(msg)
    raise TimedOutException('Timed out waiting for {desc} (timeout={timeout})'.format(desc=action_desc, timeout=reboot_timeout))