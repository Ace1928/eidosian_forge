import six
import sys
import time
import traceback
import random
import asyncio
import functools
def random_sleep(self, previous_attempt_number, delay_since_first_attempt_ms):
    """Sleep a random amount of time between wait_random_min and wait_random_max"""
    return random.randint(self._wait_random_min, self._wait_random_max)