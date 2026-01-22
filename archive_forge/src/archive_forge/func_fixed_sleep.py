import six
import sys
import time
import traceback
import random
import asyncio
import functools
def fixed_sleep(self, previous_attempt_number, delay_since_first_attempt_ms):
    """Sleep a fixed amount of time between each retry."""
    return self._wait_fixed