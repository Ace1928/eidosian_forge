import random
import time
import six
@property
def total_attempts(self):
    """The total amount of backoff attempts that will be made."""
    return self._total_attempts