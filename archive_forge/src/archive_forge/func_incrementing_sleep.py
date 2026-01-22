import six
import sys
import time
import traceback
import random
import asyncio
import functools
def incrementing_sleep(self, previous_attempt_number, delay_since_first_attempt_ms):
    """
        Sleep an incremental amount of time after each attempt, starting at
        wait_incrementing_start and incrementing by wait_incrementing_increment
        """
    result = self._wait_incrementing_start + self._wait_incrementing_increment * (previous_attempt_number - 1)
    result = min(result, self._wait_incrementing_max)
    result = max(result, 0)
    return result