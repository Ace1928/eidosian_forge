import six
import sys
import time
import traceback
import random
import asyncio
import functools
def exponential_sleep(self, previous_attempt_number, delay_since_first_attempt_ms):
    exp = 2 ** previous_attempt_number
    result = self._wait_exponential_multiplier * exp
    result = min(result, self._wait_exponential_max)
    result = max(result, 0)
    return result