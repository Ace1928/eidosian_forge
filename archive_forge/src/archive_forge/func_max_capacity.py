import threading
import time
from botocore.exceptions import CapacityNotAvailableError
@property
def max_capacity(self):
    return self._max_capacity