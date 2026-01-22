import sys
import datetime
from collections import namedtuple
def whitelist(self, keys):
    """Report the corresponding key as ham, incrementing the whitelist
        count.

        Engines that implement don't implement this method should have
        handles_one_step set to False.
        """
    raise NotImplementedError()