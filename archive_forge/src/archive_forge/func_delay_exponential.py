import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
def delay_exponential(base, growth_factor, attempts):
    """Calculate time to sleep based on exponential function.

    The format is::

        base * growth_factor ^ (attempts - 1)

    If ``base`` is set to 'rand' then a random number between
    0 and 1 will be used as the base.
    Base must be greater than 0, otherwise a ValueError will be
    raised.

    """
    if base == 'rand':
        base = random.random()
    elif base <= 0:
        raise ValueError(f"The 'base' param must be greater than 0, got: {base}")
    time_to_sleep = base * growth_factor ** (attempts - 1)
    return time_to_sleep