from __future__ import (absolute_import, division, print_function)
import re
import time
import traceback
from ansible.module_utils.basic import (AnsibleModule, env_fallback,
from ansible.module_utils.common.text.converters import to_text
def wait_to_finish(target, pending, refresh, timeout, min_interval=1, delay=3):
    is_last_time = False
    not_found_times = 0
    wait = 0
    time.sleep(delay)
    end = time.time() + timeout
    while not is_last_time:
        if time.time() > end:
            is_last_time = True
        obj, status = refresh()
        if obj is None:
            not_found_times += 1
            if not_found_times > 10:
                raise HwcModuleException('not found the object for %d times' % not_found_times)
        else:
            not_found_times = 0
            if status in target:
                return obj
            if pending and status not in pending:
                raise HwcModuleException('unexpected status(%s) occurred' % status)
        if not is_last_time:
            wait *= 2
            if wait < min_interval:
                wait = min_interval
            elif wait > 10:
                wait = 10
            time.sleep(wait)
    raise HwcModuleException('async wait timeout after %d seconds' % timeout)