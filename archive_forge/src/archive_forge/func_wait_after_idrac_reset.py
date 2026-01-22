from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def wait_after_idrac_reset(idrac, wait_time_sec, interval=30):
    time.sleep(interval // 2)
    msg = RESET_UNTRACK
    wait = wait_time_sec
    track_failed = True
    while wait > 0:
        try:
            idrac.invoke_request(MANAGERS_URI, 'GET')
            time.sleep(interval // 2)
            msg = RESET_SUCCESS
            track_failed = False
            break
        except Exception:
            time.sleep(interval)
            wait = wait - interval
    return (track_failed, msg)