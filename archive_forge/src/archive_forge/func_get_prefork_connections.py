import sys
import datetime
from collections import namedtuple
@classmethod
def get_prefork_connections(cls, fn, mode, max_age=None):
    """Yields an unlimited number of partial functions that return a new
        engine instance, suitable for using toghether with the Pre-Fork server.
        """
    raise NotImplementedError()