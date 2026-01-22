import six
import sys
import time
import traceback
import random
import asyncio
import functools
@staticmethod
def no_sleep(previous_attempt_number, delay_since_first_attempt_ms):
    """Don't sleep at all before retrying."""
    return 0