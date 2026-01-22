from .util import (
import sys
from functools import reduce
def unuse_region(self):
    """Unuse the current region. Does nothing if we have no current region

        **Note:** the cursor unuses the region automatically upon destruction. It is recommended
        to un-use the region once you are done reading from it in persistent cursors as it
        helps to free up resource more quickly"""
    if self._region is not None:
        self._region.increment_client_count(-1)
    self._region = None