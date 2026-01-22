from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.cloud import CloudRetry
import random
from functools import wraps
import syslog
import time
def backoff_gen():
    for retry in range(0, retries):
        yield _random.randint(0, min(max_delay, delay * 2 ** retry))