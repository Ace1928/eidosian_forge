import random
import re
import time
from heat.common.i18n import _
def retry_backoff_delay(attempt, scale_factor=1.0, jitter_max=0.0):
    """Calculate an exponential backoff delay with jitter.

    Delay is calculated as
    2^attempt + (uniform random from [0,1) * jitter_max)

    :param attempt: The count of the current retry attempt
    :param scale_factor: Multiplier to scale the exponential delay by
    :param jitter_max: Maximum of random seconds to add to the delay
    :returns: Seconds since epoch to wait until
    """
    exp = float(2 ** attempt) * float(scale_factor)
    if jitter_max == 0.0:
        return exp
    return exp + random.random() * jitter_max