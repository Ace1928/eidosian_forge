import abc
import logging
from . import display
@property
def need_sort_by_cliff(self):
    """Whether sort procedure is performed by cliff itself.

        Should be overridden (return False) when there is a need to implement
        custom sorting procedure or data is already sorted.
        """
    return True